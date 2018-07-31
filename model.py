import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Encode(nn.Module):
    '''
    This is the encoder for the head movement predictor.  Previous sequence of head
    movements will be fed into the network, and a context vector will be produced
    which will be fed into the decoder network to predict the future head movements
    '''
    def __init__(self, hidden_dim, input_dim, num_lstm_layers, batch_size):
        #hidden_dim is dimension of hidden layer of LSTM cell(s)
        #input_dim is the size of one input (3rd dimension of an 'example')
        #num_lstm_layers is the number of stacked LSTM cells
        #batch_size is exactly that: batch_size of inputs
        super(Encode, self).__init__()
        self.num_lstm_layers = num_lstm_layers
        self.batch_size = batch_size
        self.hidden_layer_dimension = hidden_dim
        #nn.LSTM(input_dimension, hidden_dimension)
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.hidden = self.init_hidden()
        self.first = True
        self.input_dim = input_dim

    def forward(self, example):
        #example expects data to be a [length of seq x batch size x data_size] tensor
        #check for input dimensions
        if self.first:
            if example.shape[1] != self.batch_size:
                print('ERROR: dimension[1] = (%i) of training example != batch_size = (%i)' %(example.shape[1], self.batch_size))
                exit()
            elif example.shape[2] != self.input_dim:
                print('ERROR: dimension[2] = (%i) of training example != input_dim = (%i)' %(example.shape[2], self.input_dim))
                exit()
            else:
                print('encoder input dimensionality check complete')
            self.first = False

        lstm_out, self.hidden = self.lstm(example, self.hidden)
        return lstm_out, self.hidden

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.num_lstm_layers, self.batch_size, self.hidden_layer_dimension).cuda(), torch.zeros(self.num_lstm_layers, self.batch_size, self.hidden_layer_dimension).cuda())




class Decode(nn.Module):
    '''
    This is the decoder for the head movement predictor.  The encoder network will
    produce a context vector which is fed into the hidden state of the decoder, producing a new
    sequence of describing the
        - probability that a bin is in the user's field of view
    '''
    def __init__(self, hidden_size, output_size, num_lstm_layers, batch_size, droupout_p = 0.1, max_length=500):
        #hidden_size is the size of the last hidden state presented by the encoder (int)
        #output_size is the size of the output for each step of this LSTM (int)
            #NOTE: output size is not determined yet becuase not sure if it will be a
            #quaternion representation or 3D vector or bin representation
            #NOTE: currently, output_size will be the total number of bins, with
            #a log_softmax layer to determine the probability of each bin being viewed
        #num_lstm_layers is the number of stacked LSTM cells
        #batch_size is exactly that: batch_size of inputs
        #droupout is the probability that a feature is dropped and made zero
        #max_length is the maximum length of the decoded sequence, in this case
        #the number of frames wanted to be predicted
        super(Decode, self).__init__()
        self.num_lstm_layers = num_lstm_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.droupout_p = droupout_p
        self.max_length = max_length
        self.batch_size = batch_size
        #self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        #self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.droupout_p)
        self.lstm = nn.LSTM(self.output_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim = -1)
        self.hidden = self.initHidden()
        self.first = 0

    def forward(self, input, hidden):
        #dimensionality checks
        if self.first < 2:
            if input.shape[2] != self.output_size:
                print('ERROR: dimension[0] = (%i) of input vector != output_size = (%i)' %(input.shape[2], self.output_size))
                exit()
            elif input.shape[1] != self.batch_size:
                print('ERROR: dimension[1] = (%i) of input vector != batch_size = (%i)' %(input.shape[1], self.batch_size))
                exit()
            elif hidden[0].shape[2] != self.hidden_size:
                print('ERROR: dimension[0] = (%i) of hidden vector != hidden_size = (%i)' %(hidden[0].shape[2], self.hidden_size))
                exit()
            elif hidden[0].shape[1] != self.batch_size:
                print('ERROR: dimension[1] = (%i) of hidden vector != batch_size = (%i)' %(hidden[0].shape[1], self.batch_size))
                exit()
            else:
                print('decoder input dimensionality check #%i complete' %(self.first + 1))
            self.first = 1 + self.first

        #input is expected to be the <SOS> vector, or previous output produced from
        #the decoder, with shape [num_bins x batch_size]
        #hidden is expected to be the context vector from the encoder, or the last hidden state from
        #of the decoder, with shape [encoder_output_size, batch_size]
        input = input.view(1, self.batch_size, -1)
        input = self.dropout(input)
        input = F.relu(input)
        lstm_out, hidden = self.lstm(input.float(), hidden)
        #lstm_out is [seq_len(is 1) x batch_size x hidden_dim], so we only want the first output
        output = self.out(lstm_out[0].float())
        output = self.softmax(output)
        return output, hidden


    def initHidden(self):
        return (torch.zeros(self.num_lstm_layers, self.batch_size, self.hidden_size).cuda(), torch.zeros(self.num_lstm_layers, self.batch_size, self.hidden_size).cuda())
