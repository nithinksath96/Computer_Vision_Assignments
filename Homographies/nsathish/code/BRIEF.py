import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from keypointDetect import DoGdetector
from scipy.interpolate import interp1d

def makeTestPattern(patch_width=9, nbits=256):
    '''
    Creates Test Pattern for BRIEF
    Run this routine for the given parameters patch_width = 9 and n = 256

    INPUTS
        patch_width - the width of the image patch (usually 9)
        nbits       - the number of tests n in the BRIEF descriptor

    OUTPUTS
        compareX and compareY - LINEAR indices into the patch_width x patch_width image 
                                patch and are each (nbits,) vectors. 
    '''
    
    #############################
    # TO DO ...
    # Generate testpattern here
    
    np.random.seed(0)

    compareX=np.random.normal(0,(patch_width**2)/25,nbits)
    compareY=np.random.normal(0,(patch_width**2)/25,nbits)
    
    #print("CompareX",compareX)
    #print("CompareY",compareY)
    X_Y=[compareX, compareY]
    np.save("../results/testPattern.npy",X_Y)
    
    return  compareX, compareY


# load test pattern for Brief
#test_pattern_file = '../results/testPattern.npy'
#if os.path.isfile(test_pattern_file):
#    # load from file if exists
#    compareX, compareY = np.load(test_pattern_file)
#else:
#    # produce and save patterns if not exist
#    compareX, compareY = makeTestPattern()
#    if not os.path.isdir('../results'):
#        os.mkdir('../results')
#    np.save(test_pattern_file, [compareX, compareY])


def computeBrief(im, gaussian_pyramid, locsDoG, k, levels,
    compareX, compareY):
    '''
    Compute brief feature
    INPUT
        locsDoG - locsDoG are the keypoint locations returned by the DoG
                detector.
        levels  - Gaussian scale levels that were given in Section1.
        compareX and compareY - linear indices into the 
                                (patch_width x patch_width) image patch and are
                                each (nbits,) vectors.
    
    
    OUTPUT
        locs - an m x 3 vector, where the first two columns are the image
                coordinates of keypoints and the third column is the pyramid
                level of the keypoints.
        desc - an m x n bits matrix of stacked BRIEF descriptors. m is the number
                of valid descriptors in the image and will vary.
    '''
    
    ##############################
    # TO DO ...
    # compute locs, desc here

    patch_size=9
    nbits=256
    locs=np.zeros((1,3))
    desc=np.zeros((1,nbits))
    for i in range(locsDoG.shape[0]):
        center = [locsDoG[i][0],locsDoG[i][1]]
        tl_x= int(center[0] -patch_size/2)
        tl_y= int(center[1] -patch_size/2)
        br_x =int(center[0] + patch_size/2)
        br_y = int(center[1] + patch_size/2)
        
        
        #print("TLX",patch_size)
        if(tl_x >= 0 and tl_y >= 0 and br_x <= im.shape[0] and br_y <= im.shape[1]):
            patch = gaussian_pyramid[tl_x : tl_x + patch_size, tl_y : tl_y + patch_size,locsDoG[i][2]].flatten()
            #print("Patch",patch.shape)
            compareX = compareX + center[0]
            compareY = compareY + center[1]
            
            min_x =np.min(compareX)
            max_x=np.max(compareX)
            min_y =np.min(compareY)
            max_y=np.max(compareY)
            
            interpx =interp1d([min_x,max_x],[0,80])
            interpy =interp1d([min_y,max_y],[0,80])
            
            compareX=interpx(compareX)
            compareY=interpy(compareY)
            
            compareX_int = compareX.astype(int)
            compareY_int = compareY.astype(int)
            #print("center",center)
            #print("MinX",np.min(compareX_int))
            #print("MinY",np.min(compareY_int))
            if(desc.shape[0]==1):
                desc = 1*(patch[compareX_int] < patch[compareY_int])
                locs = np.array([center[0],center[1], locsDoG[i][2]])
                #print("des",des.shape)
            else:
                #print("new",(1*(patch[compareX_int] < patch[compareY_int])).shape)
                
                desc = np.vstack((desc,1*(patch[compareX_int] < patch[compareY_int])))
                locs =np.vstack((locs,np.array([center[0],center[1], locsDoG[i][2]])))
    
    
    return locs, desc


def briefLite(im):
    '''
    INPUTS
        im - gray image with values between 0 and 1

    OUTPUTS
        locs - an m x 3 vector, where the first two columns are the image coordinates 
            of keypoints and the third column is the pyramid level of the keypoints
        desc - an m x n bits matrix of stacked BRIEF descriptors. 
            m is the number of valid descriptors in the image and will vary
            n is the number of bits for the BRIEF descriptor
    '''
    
    ###################
    # TO DO ...
    levels = [-1,0,1,2,3,4]
    k=np.sqrt(2)
    locsDoG, gaussian_pyramid = DoGdetector(im)
    test_pattern_file = '../results/testPattern.npy'
    compareX , compareY = np.load(test_pattern_file)
    locs,desc=computeBrief(im, gaussian_pyramid, locsDoG, k, levels,compareX, compareY)
    #print(desc.shape)
    #print(locs.shape)
    #print(compareX.shape)
    #print(compareY.shape)
    #print(locsDoG.shape)
    #print(gaussian_pyramid.shape)
    
    
    
    return locs, desc


def briefMatch(desc1, desc2, ratio=0.8):
    '''
    performs the descriptor matching
    INPUTS
        desc1, desc2 - m1 x n and m2 x n matrix. m1 and m2 are the number of keypoints in image 1 and 2.
                                n is the number of bits in the brief
    OUTPUTS
        matches - p x 2 matrix. where the first column are indices
                                        into desc1 and the second column are indices into desc2
    '''
    
    D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')
    # find smallest distance
    ix2 = np.argmin(D, axis=1)
    d1 = D.min(1)
    # find second smallest distance
    d12 = np.partition(D, 2, axis=1)[:,0:2]
    d2 = d12.max(1)
    r = d1/(d2+1e-10)
    is_discr = r<ratio
    ix2 = ix2[is_discr]
    ix1 = np.arange(D.shape[0])[is_discr]

    matches = np.stack((ix1,ix2), axis=-1)
    return matches


def plotMatches(im1, im2, matches, locs1, locs2):
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')
    
    locs1[:,[0, 1]] = locs1[:,[1, 0]]
    locs2[:,[0, 1]] = locs2[:,[1, 0]]
    
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i,0], 0:2]
        pt2 = locs2[matches[i,1], 0:2].copy()
        pt2[0] += im1.shape[1]
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x,y,'r')
        plt.plot(x,y,'g.')
    plt.show()    
    

if __name__ == '__main__':
    # test makeTestPattern
    #compareX, compareY = makeTestPattern()
    
    # test briefLite
#    im = cv2.imread('../data/model_chickenbroth.jpg')
#    locs, desc = briefLite(im)  
#    fig = plt.figure()
#    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cmap='gray')
#    plt.plot(locs[:,0], locs[:,1], 'r.')
#    plt.draw()
#    plt.waitforbuttonpress(0)
#    plt.close(fig)
#    
#    # test matches
    im1 = cv2.imread('../data/pf_scan_scaled.jpg')
    im2 = cv2.imread('../data/pf_pile.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
#    locs1[:][0] ,locs1[:][1] = locs1[:][1] ,locs1[:][0]
#    locs2[:][0] ,locs2[:][1] = locs2[:][1] ,locs2[:][0]
    #plotMatches(im1,im2,matches,locs1,locs2)