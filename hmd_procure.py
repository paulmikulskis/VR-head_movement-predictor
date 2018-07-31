import os
from os import listdir
from os.path import isfile, join

#classes to structurally represent the data
class Participant:
    def __init__(self, uid, sex, age):
        self.sex = sex
        self.uid = uid
        self.age = age
        self.videos = []

class VideoData:
    def __init__(self, name, offset):
        self.name = name
        self.offset = offset
        self.videoData = []



class HMDGrabber:

    def numParticipants(self):
        return self.folderCount

    def get_immediate_subdirectories(self, a_dir):
        return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
                if os.path.isdir(os.path.join(a_dir, name))]

    def __init__(self, test):
        if test == 'train':
            self.dirs = self.get_immediate_subdirectories('./vr_hmd_train/results/')
        elif test == 'test':
            self.dirs = self.get_immediate_subdirectories('./vr_hmd_test/results/')
        else:
            print('specify test or train dataset to load HMD')

        self.folderCount = len(self.dirs)

    def grabData(self):
        participants = []
        #x is a list representing all the UID-xxx folders
        for x in self.dirs:
            f = open(os.path.join(x,'formAnswers.txt'),'r')
            vid_off = [join(os.path.join(x,'test0'),f) for f in listdir(os.path.join(x,'test0')) if isfile(join(os.path.join(x,'test0'), f))]
            vid_off = [t for t in vid_off if 'testInfo.txt' not in t]
            se = ""
            ag = ""
            ui = ""
            for i, line in enumerate(f):
                if i == 0:
                    ui = f.readline()[2:]
                    se = f.readline()[2:]
                    ag = f.readline()[2:]
            p = Participant(ui,se,ag)
            vidirs = self.get_immediate_subdirectories(os.path.join(x, 'test0'))
            for y in vidirs:
                #gets the singular text file for this video in the UID folder
                onlyfiles = [f for f in listdir(y) if isfile(join(y, f))]
                a = onlyfiles[0]
                #gets the NAME of the video in question and opens the file
                name = a[0:a.find('-')]
                if name[0] == '.':
                    name = name[1:]
                    print('potential error for participant %s, filename = %s' %(ui, name))
                data = open(os.path.join(y, a))
                vid_off2 = [t[t.find('test0') + 6:] for t in vid_off]
                vid_off2 = [t[:t.find('-')] for t in vid_off2]
                i = (vid_off2.index(name))
                offset = open(vid_off[i])
                for d, line in enumerate(offset):
                    if d == 9:
                        #OFFSET of video in seconds
                        off = line[line.find('=')+1:]
                vdata = VideoData(name, off)
                #procuring the data
                frame = -1
                for line in data:
                    l = line.split()
                    if frame != l[1]:
                        frame = l[1]
                        vdata.videoData.append({'time':l[0], 'frame':l[1], 'pitch':l[2], 'yaw':l[3], 'roll':l[4], 'z':l[5]})
                p.videos.append(vdata)
            participants.append(p)
        return participants
