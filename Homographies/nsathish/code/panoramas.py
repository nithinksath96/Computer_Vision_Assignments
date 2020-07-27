import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches
from numpy.linalg import inv


def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given homography matrix. 
    Warps img2 into img1 reference frame using the provided warpH() function

    INPUT
        im1 and im2 - two images for stitching
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    
    #######################################
    # TO DO ...
    #print(im1.shape)
    #warped_image=cv2.warpPerspective(im2, H2to1, (im1.shape[1],im1.shape[0]))
    warped_image=cv2.warpPerspective(im2, H2to1, (1400,600))
    #print(H2to1)

    padded_image=np.pad(im1,((0,100),(0,2000),(0,0)),mode='constant',constant_values=0)[:600,0:1400]

    pano_im=np.maximum(warped_image,padded_image)
    #np.save("../results/q6_1.npy", H2to1)
    #cv2.imshow("warped",warped_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given homography matrix without clipping. 
    Warps img2 into img1 reference frame using the provided warpH() function

    INPUT
        im1 and im2 - two images for stitching
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    
    ######################################
    # TO DO ...
    sx=0.5
    sy=0.5
    tx=0
    ty=100
    M= np.array([[sx,0,tx],
                 [0,sy,ty],
                 [0,0,1]])
    warp1=cv2.warpPerspective(im1, M, (900,400))
    warp2=cv2.warpPerspective(im2, np.matmul(M,H2to1), (900,400))
    pano_im=np.maximum(warp1,warp2)
    

    cv2.imshow("Blended",blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #print(H2to1.shape)
    

    
    return pano_im


def generatePanaroma(im1, im2):
    '''
    Generate and save panorama of im1 and im2.

    INPUT
        im1 and im2 - two images for stitching
    OUTPUT
        Blends img1 and warped img2 (with no clipping) 
        and saves the panorama image.
    '''

    ######################################
    # TO DO ...
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    
    locs1[:,[0, 1]] = locs1[:,[1, 0]]
    locs2[:,[0, 1]] = locs2[:,[1, 0]]

    bestH=ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    #print(bestH)
    output=imageStitching(im1,im2,bestH)
    #output=imageStitching_noClip(im1,im2,bestH)



    return output

if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    generatePanaroma(im1, im2)