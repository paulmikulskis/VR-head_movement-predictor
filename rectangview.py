'''
This script will take video data from video 'name', and display a
video rendering where people are looking in an equirecctangular
projection of the video at hand.
Binning is done as specified in the bins variable, convention: [row, col]
load_data_set specifies 'train' or 'test'
resize is the inverse scaling factor for the video display
binIDs if set to True will display each bin's ID on the video
'''
#SETTINGS ------------------
name = "Rollercoaster"
bins = [5,8]
load_data_set = 'train'
resize = 8
binIDs = True
#---------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import cv2
import hmd_procure
import translate

#a = {'a':1, 'b':2, 'c':3, 'd': 4}
names = {'Rollercoaster': '../saliency/videos/8lsB-P8nGSM.mkv', 'Elephant': '../saliency/videos/2bpICIClAIg.webm',
'Diving': '../saliency/videos/2OzlksZBTiA.mkv', 'Rhino': '../saliency/videos/7IWp875pCxQ.webm',
'Timelapse': '../saliency/videos/CIw8R8thnm8.mkv', 'Venice': '../saliency/videos/s-AJRFQuAtE.mkv',
'Paris': '../saliency/videos/sJxiPiAaB4k.mkv'}

#setup video capture
if name in names.keys():
    filename = names[name]
    print("video '%s' found at: %s" %(name, filename))
else:
    print("video name: '%s' not valid, exiting..." %name)
    exit()


#load participant data, specify 'test' or 'train' for specific data
grabber = hmd_procure.HMDGrabber(load_data_set)
participants = grabber.grabData()
print("loading %sing HMD with a total of %i folders" %(load_data_set, grabber.numParticipants()))

#from the HMDGrabber, sort out only data with the name 'name'
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
print("----------")

#amount of bins, [rows, cols]
bins = [bins[0], bins[1]]
bin_width = int(w / bins[1])
bin_heigth = int(h / bins[0])

#rollercoaster_data is a list of lists, containing dictionary entries for each frame
#each entry in rollercoaster_data is a list of participant's head movements
#positions will be a list of lists containing the binIDs for each frame for each participant
positions = []
for i, d in enumerate(rollercoaster_data):
    #seq will hold the translated locations for each frame for this current participant
    seq = []
    #d is a list of dict entries for each frame
    for dd in d:
        frame, q0, q1, q2, q3 = dd['frame'], float(dd['pitch']), float(dd['yaw']), float(dd['roll']), float(dd['z'])
        #p is [i,j,k] rotated from [1,0,0] by (q0,q1,q2,q3)
        p = translate.rotate(q0,q1,q2,q3)
        #(x,y) is the 3D point 'p' projected onto a 2D plane sized (w,h)
        x, y = translate.equiequate(p,w,h)
        #id is the binID corresponding to (x,y) on the 2D plane sized (w,h) with [row,col] bins
        id = translate.binid([x,y],bins,[h,w])
        seq.append([frame, id])
    positions.append(seq[:])

#bbs will be the sequence of movements for each participant including
# [frame number,   binID,   [x,y] position of bottom left corener of bin]
bbs = []
for p in positions:
    #bb will be the sequence of movement for participant p
    bb = []
    for d in p:
        #d[1] is the binID corresponding to where the user is looking
        #(x,y) will now hold the bottom left corener of bin number: d[1]
        x,y = translate.bin2coor(d[1], bins,[h,w])
        #append [frame,    binID,    bin_bottom_left_corner_(x,y)]
        bb.append([d[0], d[1], [x,y]])
    bbs.append(bb[:])

#create an array of all the bins, each entry being [binID, bottomLeftX, bottomLeftY]
#for plotting purposes
numRows = bins[0]
numCols = bins[1]
#bhs and bws represent bin heights and bin widths, resepctively
bhs = [(i*(h / numRows)) for i in range(numRows)]
bws = [(i*(w / numCols)) for i in range(numCols)]
bins = []
#bin indexing starts in the bottom left with binID = 0, then progresses
#left to right, and up with increasing (x,y)
for i, h in enumerate(bhs):
    for j, w in enumerate(bws):
        bins.append([(i*numCols)+j,w,h])

#each bin is assigned an plotting value for graphing purposes (mask)
alphas = [1 for i in range(len(bins))]

#global variables for the bin height and width for drawing the rect
binx = bin_width
biny = bin_heigth
#print('binx: %i, biny: %i' %(binx,biny))
#fig, ax = plt.subplots(1,1)
plt.ion()
fig = plt.figure(figsize=(16, 9))
plt.show()

#main display loop, will show the entire video
for i in range(frames):
    #reset alpha values to all black
    alphas = [1 for a in alphas]
    #clear figure
    fig.clf()
    #get current axis to set title, set limits, and create tick marks
    ax = plt.gca()
    ax.set_title('gaze direction map')
    ax.set_xlim(0, w + 1)
    ax.set_ylim(0, h + 1)
    ax.set_xticks(np.arange(0, int(w) + bin_width + 1, bin_width))
    ax.set_yticks(np.arange(0, int(h) + bin_heigth + 1, bin_heigth))
    #get the frame and reseize it, then flip it (reads upside down)
    flag, frame = cap.read()
    frame = cv2.resize(frame, (int(w) + bin_width + 1, int(h) + bin_heigth + 1))
    frame = cv2.flip(frame,0)
    #bbs will be the sequence of movements for each participant including
    # [frame number,   binID,   [x,y] position of bottom left corener of bin]
    #as long as frame 'i' is present in the array 'b', then modify the alpha value
    #for frame i for binID = b[i][1] by 10%
    for b in bbs:
        if i < len(b):
            alphas[b[i][1]] -= 5/len(bbs)
            #if the alpha value is very small, then set it to zero
            if alphas[b[i][1]] < 0:
                alphas[b[i][1]] = 0

    for i, b in enumerate(bins):
        #create a bin mas for every bin at (x = b[1], y = b[2]) with width = binx, height = biny
        rect = matplotlib.patches.Rectangle((b[1],b[2]), binx, biny, fill = True, alpha = alphas[i], color = 'black')
        ax.add_patch(rect)
        if binIDs == True:
            ax.text(int(b[1] + binx*0.46), int(b[2] + biny*0.5), str(i), color = 'red')
    #create the grid to display bin boundaries
    plt.grid(which = 'major')
    #show the frame, and convert the color from BGR -> RGB for matplotlib
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #pause is necesarry for loop to work correctly
    plt.pause(0.00001)
