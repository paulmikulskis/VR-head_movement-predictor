import torch
import numpy as np
import translate
import data
import hmd_procure
import cv2
import h5py as h5

#Parameters ---------------------------------
ENCODER_PATH = 'encoder.pt'
DECODER_PATH = 'decoder.pt'
FRAMES = 300
TRIALS = 1 #how many data passes
#File options ------------------
name = "Rollercoaster"
bins = [4,8]
load_data_set = 'train'
resize = 8
binIDs = True
#--------------------------------------------
encoder = torch.load(ENCODER_PATH)
print('encoder sucesfully loaded from', ENCODER_PATH)
decoder = torch.load(DECODER_PATH)
print('decoder sucesfully loaded from', DECODER_PATH)

names = {'Rollercoaster': '../saliency/videos/8lsB-P8nGSM.mkv', 'Elephant': '../saliency/videos/2bpICIClAIg.webm',
'Diving': '../saliency/videos/2OzlksZBTiA.mkv', 'Rhino': '../saliency/videos/7IWp875pCxQ.webm',
'Timelapse': '../saliency/videos/CIw8R8thnm8.mkv', 'Venice': '../saliency/videos/s-AJRFQuAtE.mkv',
'Paris': '../saliency/videos/sJxiPiAaB4k.mkv'}

#LSTM Parameters -----------------------------
BATCH_SIZE = encoder.batch_size
ENCODER_HIDDEN_SIZE = encoder.hidden_layer_dimension
DECODER_HIDDEN_SIZE = decoder.hidden_size

hdFile = h5.File("../saliency/" + name.lower() + '.h5', 'r')
if hdFile is None:
    print("error opening hdf5 file...check path")
    exit()
else:
    print("HDF5 file '%s' sucesfully obtained" %name)


#setup video capture
if name in names.keys():
    filename = names[name]
    print("video '%s' found at: %s" %(name, filename))
else:
    print("video name: '%s' not valid, exiting..." %name)
    exit()

key = list(hdFile.keys())[0]
dataset = hdFile[key]
nbF = dataset.shape[2]

#load participant data, specify 'test' or 'train' for specific data
grabber = hmd_procure.HMDGrabber(load_data_set)
participants = grabber.grabData()
print("loading %sing HMD with a total of %i folders" %(load_data_set, grabber.numParticipants()))

#from the HMDGrabber, sort out only data with the name 'name'
offset = 0
rollercoaster_data = []
for p in participants:
    v = [v for v in p.videos if v.name == name][0]
    offset = int(v.offset)
    v = v.videoData
    #print(len(v))
    rollercoaster_data.append(v[:])

cap = cv2.VideoCapture(filename)
#set start time in ms
cap.set(cv2.CAP_PROP_POS_MSEC,offset*1000)
if not cap.isOpened():
    print("unable to open", filename)
    exit()
print("video '%s' sucessfully opened" %name)
#get the number of frames in the video, the width, and height of frames
frames = int(cap.get(7))
w  = int(cap.get(3) / resize)
h = int(cap.get(4) / resize)
print("total frames: %i, width: %i, height: %i" %(frames,w,h))

#amount of bins, [rows, cols]
bins = [bins[0], bins[1]]
bin_width = int(w / bins[1])
bin_heigth = int(h / bins[0])

dataloader = data.ShortWindowDataLoader(hdFile, participants, FRAMES, FRAMES, BATCH_SIZE, name)
SOS_token = np.ones((1, BATCH_SIZE, bins[0] * bins[1])) * -1

for i in range(TRIALS):
    print('-----------------')
    data = dataloader[iter]
    movement_past = data['hmd_prev']
    movement_future = data['hmd_fut']
    saliency_prev = data['saliency_prev']

    binIDs_fut = translate.convertHMDarrayToBinIDs(movement_future, bins, [w,h])
    saliency_prev = np.reshape(saliency_prev, (saliency_prev.shape[2], -1))
    saliency_prev = np.expand_dims(saliency_prev, 1)
    spc = saliency_prev
    for i in range(movement_past.shape[1] - 1):
        saliency_prev = np.concatenate((saliency_prev,spc),1)
    input_data = np.concatenate((movement_past, saliency_prev), 2)
    target_length = input_data.shape[0]
    input_data = torch.tensor(input_data).float().cuda()
    encoder_outputs, encoder_hidden = encoder(input_data)
    decoder_hidden = encoder_hidden
    decoder_input = torch.tensor(SOS_token).cuda()
    #use predictions as next input
    for i in range(target_length):
        #print('target length processed %i' %i)
        decoder_input, decoder_hidden = decoder(decoder_input, decoder_hidden)
        decoder_input = decoder_input.view(1, BATCH_SIZE, -1).detach()
        print('%i - predict bin %i, actual bin %i' %(i,torch.topk(decoder_input,1)[1].item(), np.argmax(binIDs_fut[i],axis=1)[0] ))
