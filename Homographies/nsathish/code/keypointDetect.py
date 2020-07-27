import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt


def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid


def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    INPUTS
        gaussian_pyramid - A matrix of grayscale images of size
                            [imH, imW, len(levels)]
        levels           - the levels of the pyramid where the blur at each level is
                            outputs

    OUTPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
        DoG_levels  - all but the very first item from the levels vector
    '''
    
    DoG_pyramid = []
    ################
    # TO DO ...
    # compute DoG_pyramid here
    
    DoG_pyramid = np.diff(gaussian_pyramid,axis=2)
    
    DoG_levels = levels[1:]
    return DoG_pyramid, DoG_levels


def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = None
    
    ##################
    # TO DO ...
    # Compute principal curvature here
    #print("Difference",-DoG_pyramid[0][0][0] + DoG_pyramid[0][2][0] + -2*DoG_pyramid[1][0][0] + 2*DoG_pyramid[1][2][0] + -DoG_pyramid[2][0][0] + DoG_pyramid[2][2][0])
    
    eps=1e-8
    dx=cv2.Sobel(DoG_pyramid,cv2.CV_64F,1,0,ksize=5)
    dy= cv2.Sobel(DoG_pyramid,cv2.CV_64F,0,1,ksize=5)
    dxx= cv2.Sobel(dx,cv2.CV_64F,1,0, ksize=5)
    dxy =cv2.Sobel(dx,cv2.CV_64F,0,1, ksize=5)
    dyx= cv2.Sobel(dy,cv2.CV_64F,1,0, ksize=5)
    dyy =cv2.Sobel(dy,cv2.CV_64F,0,1, ksize=5)
    trace_H = (dxx + dyy)**2
    det_H= np.multiply(dxx,dyy) - np.multiply(dxy,dyx)
    
    principal_curvature = trace_H / ((det_H) + eps)
    
    #print("Principal curvature",principal_curvature.shape)
    #print("img",DoG_pyramid.shape)
    #cv2.imshow("Principal Curvature", principal_curvature)
    #cv2.waitKey(0)
    
    
    #for i in range(DoG_pyramid.shape[2]):
        #print("Original",DoG_pyramid[:,:,i].shape)
        
        #dx= cv2.Sobel(DoG_pyramid[:,:,i],cv2.CV_64F,1,0)
        #print("DX",dx)

        #cv2.imshow("DXX",dx)
        #cv2.waitKey(0)
        #break
        
    
    return principal_curvature


def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = None
    
    ##############
    #  TO DO ...
    # Compute locsDoG here
    max_filter = np.array([[[0,1,0],[0,1,0],[0,1,0]],
                           [[0,1,0],[1,1,1],[0,1,0]],
                           [[0,1,0],[0,1,0],[0,1,0]]])
 
    max_vals=ndimage.maximum_filter(DoG_pyramid, footprint= max_filter)
    
    #min_vals=ndimage.minimum_filter(DoG_pyramid, footprint= max_filter)

    
    
    local_extremum = DoG_pyramid *(DoG_pyramid==max_vals)
    
    #local_extremum2 = DoG_pyramid *(DoG_pyramid==min_vals)
    
    #print(np.array((np.nonzero(local_extremum2 > th_contrast))))
    #local_extremum  = local_extremum1 + local_extremum2
   
    thresh1_result=local_extremum*(local_extremum > th_contrast)

    #print(np.array((np.nonzero(thresh1_result))))
    
    thresh2_result = principal_curvature*(principal_curvature < th_r)

    locsDoG1=np.array((np.nonzero(thresh1_result * thresh2_result)))

    locsDoG=np.transpose(locsDoG1)
    
    #locsDoG[:,[0, 1]] = locsDoG[:,[1, 0]]
    
    #print("Output",locsDoG.shape)
    
    
    
    
    
    
    
    #print(max_vals.shape)
    
    
    
    
    
    
    
    return locsDoG
  

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    INPUTS          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.


    OUTPUTS         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, locsDoG here
    
#    if len(im.shape)==3:
#        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#        
#    if im.max()>1:
#        im = np.float32(im)/255
    
    im_pyr=createGaussianPyramid(im)
    
    DoG_pyr, DoG_levels=createDoGPyramid(im_pyr, levels)
    
    pc_curvature=computePrincipalCurvature(DoG_pyr)
    
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    
    radius = 1
    #print(locsDoG[:,0])
    
    #print("Before",locsDoG)
    #locsDoG[0,:] , locsDoG[1,:] = locsDoG[1,:] , locsDoG[0,:]
    #locsDoG[:,[0, 1]] = locsDoG[:,[1, 0]]
    #print("After",locsDoG)
 #Blue color in BGR 
    color = (0, 255, 0) 
   
 #Line thickness of 2 px 
    thickness = 1
    
#    for i in range(locsDoG.shape[0]):
#    
#        output=cv2.circle(im, (locsDoG[i][1],locsDoG[i][0]),radius,color,thickness)
#
#    cv2.imshow("Output",output)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    cv2.imwrite("image.jpg",output)
    
##    
    return locsDoG, im_pyr


if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    #print("Image",im.shape)
    im_pyr = createGaussianPyramid(im)
    #print("Im pyr",im_pyr.shape)
    #displayPyramid(im_pyr)
    
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    #displayPyramid(DoG_pyr)
    
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    
    #print("pc",pc_curvature.shape)
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)