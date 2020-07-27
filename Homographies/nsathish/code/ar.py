import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from planarH import computeH
from numpy.linalg import inv


def compute_extrinsics(K, H):
    '''
    INPUTS:
        K - intrinsic parameters matrix
        H - estimated homography
    OUTPUTS:
        R - relative 3D rotation
        t - relative 3D translation
    '''

    #############################
    # TO DO ...
    
    phi_1 = np.matmul(inv(K), H)

    u, s, vh = np.linalg.svd(phi_1[:,:2])

    s_1 = np.array([[1,0],
                    [0,1],
                    [0,0]])

    R= np.matmul(np.matmul(u,s_1),vh)

    t= np.cross(R[:,0],R[:,1]).reshape(3,1)

    phi = np.hstack((R,t))


    if(np.linalg.det(phi)==-1):
        t = t*-1
        
    scale = np.sum(phi_1[:,:2] / phi[:,:2]) / 6
        
    t= phi_1[:,2]/scale
    t= np.reshape(t,(3,1))
 
    return phi, t , scale


def project_extrinsics(K, W, R, t):
    '''
    INPUTS:
        K - intrinsic parameters matrix
        W - 3D planar points of textbook
        R - relative 3D rotation
        t - relative 3D translation
    OUTPUTS:
        X - computed projected points
    '''

    #############################
    # TO DO ...
    w=np.loadtxt('../data/sphere.txt')

    
    X=np.matmul(K,np.matmul(R,w) + t)
    

    return X


if __name__ == "__main__":
    # image
    im = cv2.imread('../data/prince_book.jpeg')

    #############################
    # TO DO ...
    # perform required operations and plot sphere
    W=np.array([[0.0,18.2,18.2,0],
                [0.0,0.0,26.0,26.0],
                [0.0,0.0,0.0,0.0]])
    K=np.array([[3043.72,0.0,1196.00],
                [0.0,3043.72,1604.00],
                [0.0,0.0,1.0]])
    X=np.array([[483,1704,2175,67],
                [810,781,2217,2286]])
    

    H=computeH(W[:2,:],X)
    

 
    R,t,scale= compute_extrinsics(K,H)

    
    X=project_extrinsics(K,W,R,t)
    X = X / X[2,:]

#    plt.imshow(im)
#    for i in range(X.shape[1]):
#        plt.plot(X[0][i]+400,X[1][i]+600, 'y.', markersize=1)
#    plt.show()   

        
        