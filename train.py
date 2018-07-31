import data
import h5py as h5
import hmd_procure
import model
import time
import math
import translate
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


print('')
print("------ begin procuring head movement data ------")
#PARAMETERS -----------------------------------------------
#name of the saliency map file in '../saliency/'
hdfName = 'rollercoaster.h5'
#define parameters for tensor construction:
#number of [previous frames] used to predict the number of [future frames]
frame_window_prev = 5
frame_window_fut = 5
#if printing for validation, how many samples to print
samplePrintSize = 3
printcheck = False
MAX_LENGTH = 500 #frames
#LSTM Parameters -------------------------------------------
BINS = [4,8]
BATCH_SIZE = 1

NUM_ENCODER_LSTM_LAYERS = 1
NUM_DECODER_LSTM_LAYERS = 1

ENCODER_HIDDEN_SIZE = 200
DECODER_HIDDEN_SIZE = 200
DROUPOUT_P_DECODER = 0.01

#Training Parameters ---------------------------------------
EPOCHS = 20
PLOT = True
PLOT_EVERY = 2
PRINT_EVERY = 5
LEARNING_RATE = 0.01

#Model Save Parameters -------------------------------------
SAVE = True  #option to save the models or not
ENCODER_SAVE_NAME = 'encoder.pt'
DECODER_SAVE_NAME = 'decoder.pt'

# ---------- -----------------------------------------------
IMG_DIMS = [0,0]

SOS_token = np.ones((1, BATCH_SIZE, BINS[0] * BINS[1])) * -1
#grab the HDF5 file containing the saliency map
hdFile = h5.File("../saliency/" + hdfName, 'r')
if hdFile is None:
    print("error opening hdf5 file...check path")
    exit()
else:
    print("HDF5 file '%s' sucesfully obtained" %hdfName)

#load participant data, specify 'test' or 'train' for specific data
load = 'test'
grabber = hmd_procure.HMDGrabber(load)
participants = grabber.grabData()
print("loading %sing HMD with a total of %i folders" %(load, grabber.numParticipants()))
videoKey = hdfName[0:hdfName.find('.')]
l = videoKey[0].upper()
videoKey = videoKey[1:]
videoKey = l + videoKey
if videoKey in [name.name for name in participants[0].videos]:
    print("found key '%s' for grabbing participant data" %videoKey)
else:
    print("unable to find matching key for '%s' in grabbing pariticpant data, check HDF filename" %videoKey)
    exit()

key = list(hdFile.keys())[0]
dataset = hdFile[key]
nbF = dataset.shape[2]
dataloader = data.ShortWindowDataLoader(hdFile, participants, frame_window_prev, frame_window_fut, BATCH_SIZE, videoKey)
if printcheck:
    dataloader.datacheck()
    dataloader.printcheck(samplePrintSize)
d1 = dataloader[1]
IMG_DIMS = [d1['saliency_prev'].shape[0], d1['saliency_prev'].shape[1]]
#input dimension length = [(w*h) + 4 + 1] = [saliency_map_dims + quaternion + age_number]
in_d = d1['saliency_prev'].shape[0] * d1['saliency_prev'].shape[1] + 4 + 2
#initialize encoder and decoder objects
encoder = model.Encode(ENCODER_HIDDEN_SIZE, in_d, NUM_ENCODER_LSTM_LAYERS, BATCH_SIZE).cuda()
decoder = model.Decode(DECODER_HIDDEN_SIZE, BINS[0] * BINS[1], NUM_DECODER_LSTM_LAYERS, BATCH_SIZE, DROUPOUT_P_DECODER, MAX_LENGTH).cuda()


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    '''
    Performs one step of training for a batch of data
    INPUTS:
    - input_tensor should be the procured input example of size [seq_len x batch_size x data_size]
    - target_tensor should be the sequence of bins where a participant looks of size [seq_len x batch_size x data_size]
    - encoder and decoder are the encoder and decoder network objects
    - encoder_optimizer and decoder_optimizer are the optimizers for encoder and decoder
    - criterion is the type of loss used
    - max_length is the maximum length sequence predicted by this network
    '''
    encoder_hidden = encoder.init_hidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    #encoder_outputs = torch.zeros(max_length, encoder.hidden_layer_dimension)
    loss = 0
    #encoder_outputs holds all the hidden states in the sequence
    #encoder_hidden is (hidden_state, cell_state) of the last step of the LSTM
    encoder_outputs, encoder_hidden = encoder(input_tensor)
    decoder_hidden = encoder_hidden

    decoder_input = torch.tensor(SOS_token).cuda()
    #use predictions as next input
    for i in range(target_length):
        #print('target length processed %i' %i)
        decoder_input, decoder_hidden = decoder(decoder_input, decoder_hidden)
        idx_target = torch.topk(target_tensor[i],1)[0].long().squeeze(1).cuda()
        loss += criterion(decoder_input, idx_target)
        decoder_input = decoder_input.view(1, BATCH_SIZE, -1).detach()

    loss.backward(retain_graph=True)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, n_iters, print_every = 2, plot_every = 100, learning_rate = 0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr = learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        data = dataloader[iter]
        movement_past = data['hmd_prev']
        movement_future = data['hmd_fut']
        saliency_prev = data['saliency_prev']
        #reshaping saliency_prev into a [seq_len x batch_size x (w * h)] array:
        saliency_prev = np.reshape(saliency_prev, (saliency_prev.shape[2], -1))
        saliency_prev = np.expand_dims(saliency_prev, 1)
        spc = saliency_prev
        for i in range(movement_past.shape[1] - 1):
            saliency_prev = np.concatenate((saliency_prev,spc),1)

        target_bins = translate.convertHMDarrayToBinIDs(movement_future, BINS, IMG_DIMS)
        input_data = np.concatenate((movement_past, saliency_prev), 2)
        #tensor conversion
        input_data = torch.tensor(input_data).float().cuda()
        target_bins = torch.tensor(target_bins).float().cuda()

        loss = train(input_data, target_bins, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


    if SAVE:
        torch.save(encoder, ENCODER_SAVE_NAME)
        print('encoder model sucesfully saved as', ENCODER_SAVE_NAME)
        torch.save(decoder, DECODER_SAVE_NAME)
        print('decoder model sucesfully saved', DECODER_SAVE_NAME)

    if PLOT:
        plt.plot(plot_losses)
        plt.title('Neg Log Loss')
        plt.xlabel('epochs')
        plt.show()

trainIters(encoder, decoder, EPOCHS, PRINT_EVERY, PLOT_EVERY, LEARNING_RATE)
