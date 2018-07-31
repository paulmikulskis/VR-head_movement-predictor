#file for handling translations of coordinates
import math
import numpy as np

def H(h1,h2):
    '''computes the hamilton product of two
    hamilton quaternions, example of quaternion:
    [       O          ,           i         ,        j          ,          k        ]
    0.03693784403562278,-0.006942703182342886,0.00607127185627635,-0.9992748178914548
    '''
    a1,b1,c1,d1 = h1
    a2,b2,c2,d2 = h2
    a = a1*a2 - b1*b2 - c1*c2 - d1*d2
    b = a1*b2 + b1*a2 + c1*d2 - d1*c2
    c = a1*c2 - b1*d2 + c1*a2 + d1*b2
    d = a1*d2 + b1*c2 - c1*b2 + d1*a2
    return [a,b,c,d]

def polar2vec(theta, phi):
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)
    return [x,y,z]

def rotate(q0,q1,q2,q3,pi = 1,pj = 0,pk = 0):
    '''takes the hamilton quaternion represented
    by (q0,q1,q2,q3) and applies the representational
    rotation to the point (pi,pj,pk), and returns
    (pi',pj',pk') in the form of a triplet
    math from: math.stackexchange.com/questions/40164/how-do-you-rotate-a-vector-by-a-unit-quaternion
    '''
    #these parameters adjust the reference position if needed
    theta = math.pi * (4/8)
    phi = math.pi * (4/4)
    #phi = 0

    p = [0,pi,pj,pk]
    p = [0] + polar2vec(theta,phi)
    r = [q0,q1,q2,q3]
    rprime = [q0,-q1,-q2,-q3]
    pprime = H(H(r,p),rprime)
    #return (pprime[1:], math.sqrt(pprime[1]**2 + pprime[2]**2 + pprime[3]**2))
    return (pprime[1:])

def equiequate(v,imgdims):
    '''computes the (x,y) coordinates of the equirectangular
    projection of the 3d vector v: [i,j,k] onto a 2d plane size w x h
    '''
    i = v[0]
    j = v[1]
    k = v[2]
    w = imgdims[0]
    h = imgdims[1]
    #angle between two vectors: cos(theta) = (v1 . v2) / (|v1| * |v2|)
    #here, reference angle position is [1,0,0]
    #print('i: %f, j: %f, k: %f' %(i,j,k))
    #print('j:', j)
    neg = 1 if j > 0.0 else -1
    neg_y = 1 if k > 0.0 else -1
    yaw = 0
    pitch = 0
    if k < 1.0 and k > -1.0:
        yaw = math.acos(round((i)/(math.sqrt(math.pow(i,2) + math.pow(j,2))), 4))
        pitch = round(math.sqrt(math.pow(i,2) + math.pow(j,2)),4)
        pitch = math.asin(pitch)

        if pitch == (math.pi / 2):
            pitch = (3.14 / 2)
    else:
        print('EDGE ALERT --------------------')
        pitch = 0.0001
        yaw = 0.0001
    #print('pitch: %f, yaw: %f' %(pitch, yaw))
    pi = math.pi
    x = (neg * ((yaw) / (2*pi)) * (w)) + (w / 2)
    y = (h / 2) + (neg_y * (1 - (pitch / (pi / 2))) * (h / 2))
    if y <= 0:
        print("ALERT!!!! ON LESS THAN ZERO")
        print('y value calculated:', y)
        print('x value calculated:', x)
        print('k value calculated:', k)
        print('pitch value calculated:', pitch)
    if y >= h:
        print("ALERT!!!! ON GREATER THAN H")
        print('h value calculated:', h)
        print('k value calculated:', k)
        print('pitch value calculated:', pitch)

    #x = ((yaw + pi) / (2*pi)) * (w)
    #y = (((pi/2) - pitch) / pi) * (h)
    #cast to int for discretization
    return (int(x), int(y))

def binid(pos,bins,imgdims):
    '''returns the bin ID for the current (x,y) position
    within the current bin setting (rows,columns).  IDs are
    composed left to right, top to bottom in [0, numBins - 1]
    pos and bins and imgdims should both be 1x2 arrays or tuples
    returns the number bin the point is in, and its (x,y) position
    '''
    x = pos[0]
    y = pos[1]
    numRows = bins[0]
    numCols = bins[1]
    w = imgdims[1]
    h = imgdims[0]

    bhs = [(i*(h / numRows)) + ((h / numRows)/2) for i in range(numRows)]
    bws = [(i*(w / numCols)) + ((w / numCols)/2) for i in range(numCols)]
    bins = []
    for i, h in enumerate(bhs):
        for j, w in enumerate(bws):
            bins.append([(i*numCols)+j,w,h])

    gd = h+w
    b = 0
    bb = 0
    for i,bin in enumerate(bins):
        d = math.sqrt((bin[1] - x)**2 + (bin[2] - y)**2)
        if d < gd:
            gd = d
            b = bin[0]
            bb = bin
    #print("from %i bins, closest is bin %i at position (%i,%i)" %(len(bins), b, bb[1], bb[2]))
    return (b)

def bin2coor(b,bins,imgdims):
    '''returns the (x,y) position of the lower left corner of a bin,
    given the bin number b, the bin layout bins [row, col], and the imgdims [w,h]
    '''
    numRows = bins[0]
    numCols = bins[1]
    w = imgdims[1]
    h = imgdims[0]
    bhs = [(i*(h / numRows)) for i in range(numRows)]
    bws = [(i*(w / numCols)) for i in range(numCols)]
    bins = []
    for i, h in enumerate(bhs):
        for j, w in enumerate(bws):
            bins.append([(i*numCols)+j,w,h])
    x,y = bins[b][1],bins[b][2]
    return x,y


def convertHMDarrayToBinIDs(x,bins,imgdims):
    '''
    takes in a [seq_len x batch_size x data_size] array of HMD and converts to
    a [seq_len x batch_size x data_size] array of binIDs that correspond to data
    [entry['pitch'], entry['yaw'], entry['roll'], entry['z'], age, entry['frame']]
    '''
    #print("converting HMD_fut to binIDs:")
    #print("HMD data type:", type(x))
    #print("HMD data shape:", x.shape)
    #print("Example HMD data entry [2]:", x[2][:][:])

    def g(d):
        pp = rotate(d[0],d[1],d[2],d[3])
        (x,y) = equiequate(pp,imgdims)
        b = binid((x,y),bins,imgdims)
        c = np.zeros((bins[0] * bins[1]))
        c[b] = 1
        return c

    #this helper function takes in ONE frame of HMD data and returns the binID
    def f(d):
        return np.array(list(map(g, d)))

    return np.array(list(map(f, x)))
