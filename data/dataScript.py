#!/usr/bin/env python3
#File for procuring and concatenating data to feed into neural net
import numpy as np
import h5py as h5
import hmd_procure
from data import ShortWindowDataLoader

#print space for legibility
for i in range(5):
    print(" ")

print("------ begin procuring head movement data ------")
#PARAMETERS -----------------------------------------------
#name of the saliency map file in '../saliency/'
hdfName = 'rollercoaster.h5'
#define parameters for tensor construction:
#number of [previous frames] used to predict the number of [future frames]
frame_window_prev = 120
frame_window_fut = 120
# ---------- -----------------------------------------------


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
#nbF = dataset.shape[2]


#--------------------------------------------------------
#example usage:
display = True
if display:
    print(" ")
    print("------ data preparation validation checks ------")
    samplePrintSize = 1
    dataloader = ShortWindowDataLoader(hdFile, participants, frame_window_prev, frame_window_fut, 4, videoKey)
    sample = dataloader[2]
    print("| Type check for data:")
    print("|   -- saliency_prev: %s \n|   -- saliency_fut: %s \n|   -- hmd_prev: %s \n|   -- hmd_fut: %s\n----" %(type(sample['saliency_prev']),type(sample['saliency_fut']),type(sample['hmd_prev']),type(sample['hmd_fut'])))
    print('')
    for i in range(len(dataloader)-(len(dataloader) - samplePrintSize)):
        sample = dataloader[i]
        print("example data number %i --------" %i)
        print("HMD frame range: (%i, %i) from random range %s, participant age: %i" %(sample['hmd_prev'][0][0][5],sample['hmd_fut'][-1][0][5], sample['frame_range'], sample['hmd_fut'][0][0][4]))
        print("   --- saliency_prev shape:", sample['saliency_prev'].shape)
        print("   --- saliency_fut shape:", sample['saliency_fut'].shape)
        print("   --- hmd_prev:", sample['hmd_prev'].shape)
        print("   --- hmd_fut:", sample['hmd_fut'].shape)
