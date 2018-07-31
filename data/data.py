#File for procuring and concatenating data to feed into neural net
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random


#ShortWindowDataLoader will load sequence samples from the HDF5 saliency file specified
#in sequence lengths of frame_window_prev and frame_window_fut
#return is a dictionary:
#dict{'saliency_prev', 'saliency_fut', 'hmd_prev', 'hmd_fut', 'frame_range'}

#hdfFile should be the File object pointing to the saliency map
#particpants should be the dictionary of participant quaternion data
#pas should be the number of previous frames used in training
#fut should be the number of frames aiming to be predicted
#bs is the batch size of the number of participants' data
#key should be the videokey (essentially the name) for the video being worked on, ex:'rollercoaster'
class ShortWindowDataLoader(Dataset):
    def __init__(self, hdfFile, particpants, pas, fut, bs, key):
        self.participants = particpants
        self.hdfFile = hdfFile
        self.frame_window_prev = pas
        self.frame_window_fut = fut
        self.dataset = hdfFile[list(hdfFile.keys())[0]]
        self.nbF = self.dataset.shape[2]
        self.partBatchSize = bs
        self.key = key
        print('----\n| ShortWindowDataLoader initialized: \n|  --%i participants\n|  --frame range: [%i,%i]\n|  --batch size: %i\n----' %(len(self.participants), self.frame_window_prev, self.frame_window_fut, self.partBatchSize))
    def __len__(self):
        #the amount of training examples we can possibly have is
        #the total number of available frames divided by
        #the training example window [past_frames + future_frames]
        #--return self.nbF/(self.frame_window_prev + self.frame_window_fut)

        #or we can have random training exmaples with a theoretical limitless supply:
        return 1000

    def getHMDicts(self):
        '''
        returns the dictionary entries for only the current video beign analyzed, along with the participant's
        age in the form of a tuple.
        '''
        bs = self.partBatchSize
        #creates a list of length [batch_size] of random numbers [0 - len(dict)]
        idxs = [int(random.random() * (len(self.participants) - 1)) for i in range(bs)]
        dicts = [(int(self.participants[i].age.strip('''\n''')),
        [v for v in self.participants[i].videos if v.name == self.key]) for i in idxs]
        return dicts

    def __getitem__(self, idx):
        #currently, dataloader gets a random number location of frames, independant of idx
        pas = self.frame_window_prev
        fut = self.frame_window_fut
        dataset = self.dataset
        frameStart = int(random.random() * (self.nbF - (pas + fut) - 1))
        frameEnd = frameStart + pas + fut
        saliency_prev = dataset[:,:,frameStart:(frameStart + pas)]
        saliency_fut = dataset[:,:,(frameStart + pas):frameEnd]
        #HMD follows the following dimensions: (seq_len x batch_size x datapoint)
        #datapoints are 5 numbers: (i, j, k, theta, AgeOfParticipant, frameNumber)
        hmd_prev = np.zeros((pas, self.partBatchSize, 6))
        hmd_fut = np.zeros((fut,self.partBatchSize, 6))
        d = self.getHMDicts()
        for i in range(len(d)):
            age = d[i][0]
            #print("batch member %i, with frame start: %i, frame end: %i" %(i,frameStart,frameEnd-1))
            dictarrPrev = d[i][1][0].videoData[frameStart:frameStart + pas]
            dictarrFut = d[i][1][0].videoData[frameStart + pas: frameEnd]
            for f in range(len(dictarrPrev)):
                entry = dictarrPrev[f]
                hmd_prev[f,i,:] = [entry['pitch'], entry['yaw'], entry['roll'], entry['z'], age, entry['frame']]
            for f in range(len(dictarrFut)):
                entry = dictarrFut[f]
                hmd_fut[f,i,:] = [entry['pitch'], entry['yaw'], entry['roll'], entry['z'], age, entry['frame']]
        example = {'saliency_prev': saliency_prev, 'saliency_fut': saliency_fut, 'hmd_prev': hmd_prev, 'hmd_fut': hmd_fut, 'frame_range': (frameStart, frameEnd - 1)}
        return example

    def datacheck(self):
        print("| Type check for data:")
        print("|   -- saliency_prev: %s \n|   -- saliency_fut: %s \n|   -- hmd_prev: %s \n|   -- hmd_fut: %s\n----" %(type(self.__getitem__(2)['saliency_prev']),type(self.__getitem__(2)['saliency_fut']),type(self.__getitem__(2)['hmd_prev']),type(self.__getitem__(2)['hmd_fut'])))

    def printcheck(self, samplePrintSize):
        print("| Print check for data:")
        for i in range(self.__len__()-(self.__len__() - samplePrintSize)):
            sample = self.__getitem__(i)
            print("|  -- [example data number %i]" %i)
            print("|  | -- HMD frame range: (%i, %i) from random range %s, participant age: %i" %(sample['hmd_prev'][0][0][5],sample['hmd_fut'][-1][0][5], sample['frame_range'], sample['hmd_fut'][0][0][4]))
            print("|  | -- saliency_prev shape:", sample['saliency_prev'].shape)
            print("|  | -- saliency_fut shape:", sample['saliency_fut'].shape)
            print("|  | -- hmd_prev:", sample['hmd_prev'].shape)
            print("|  | -- hmd_fut:", sample['hmd_fut'].shape)
        print('----')
