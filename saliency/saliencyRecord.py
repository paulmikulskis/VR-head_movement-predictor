import cv2
import matplotlib.pyplot as plt
import pySaliencyMap
import tables
import numpy as np
# main
if __name__ == '__main__':
    first = True

    # set up webcams
    capture = cv2.VideoCapture('videos/8lsB-P8nGSM.mkv')
    capture.set(cv2.CAP_PROP_POS_MSEC,65000)
    # repeat until pressing a key "q"
    t = 2
    f = 2
    nbF = 0


    def rescale_frame(frame, percent = 15):
        width = int(frame.shape[1] * percent/ 100)
        height = int(frame.shape[0] * percent/ 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

    while(nbF < 2100):
        # capture
        retval, frame = capture.read()
        frame = rescale_frame(frame)
        # initialize
        frame_size = frame.shape
        frame_width  = frame_size[1]
        frame_height = frame_size[0]
        sm = pySaliencyMap.pySaliencyMap(frame_width, frame_height)


        contrast = 90

        # computation
        saliency_map = sm.SMGetSM(frame)
        saliency_map = cv2.GaussianBlur(saliency_map,(51,51),0)
        saliency_map = cv2.GaussianBlur(saliency_map,(101,101),0)

        saliency_map = np.array(saliency_map * 255, dtype = np.uint8)
        saliency_map = cv2.equalizeHist(saliency_map)
        retval, saliency_map = cv2.threshold(saliency_map, 70, 255, cv2.THRESH_TOZERO)
        saliency_map = np.array(saliency_map / 255, dtype = np.float64)

        for i in range(frame_height- int((frame_height * 0.2)), frame_height):
            saliency_map[:][i] -= ((i - (frame_height - int((frame_height * 0.2)) - 2)) / int((frame_height * 0.2)) + 0.00)
        '''
        for i in range(frame_height):
            for j in range(frame_width):
                if saliency_map[i][j] > 0.6:
                    saliency_map[i][j] += 0.2
        '''

        #img = cv2.imread(saliency_map,0)
        shape1 = saliency_map.shape[0]
        shape2 = saliency_map.shape[1]
        if first == True:
            first = False
            f = tables.open_file('rollercoaster.h5', mode='w')
            #f.root.data.remove()
            #f.remove_node(f.root, name='data', recursive=False)
            atom = tables.Float64Atom()
            t = f.create_earray(f.root, 'data', atom, (shape1, shape2, 0))
            #t = f.root.data
        saliency_map = np.expand_dims(saliency_map, axis = 2)
        f.root.data.append(saliency_map)
        print('frame complete, nbF =', nbF)
        nbF += 1
#        binarized_map = sm.SMGetBinarizedSM(frame)
#        salient_region = sm.SMGetSalientRegion(frame)
        # visualize
        #cv2.imshow('Input image', cv2.flip(frame, 1))\
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255
        vis = np.concatenate((cv2.flip(frame, 1), cv2.flip(saliency_map, 1)), axis=0)
        cv2.imshow('Saliency map', vis)
#        cv2.imshow('Binalized saliency map', cv2.flip(binarized_map, 1))
#        cv2.imshow('Salient region', cv2.flip(salient_region, 1))
        # exit if the key "q" is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
t.close
