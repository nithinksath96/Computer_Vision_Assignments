from numpy.linalg import inv
import numpy as np
import math
from scipy import interpolate

def warp(im, A, output_shape):
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation."""
    output=np.zeros((output_shape[0],output_shape[1]))
     #print(output_shape)
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
    
            des_vec=np.array([i,j,1]).reshape(3,1)
    
            source_vec=np.round(np.matmul(inv(A),des_vec))
    
            if(source_vec[0] >=200 or source_vec[0] < 0 or source_vec[1] >=150 or source_vec[1] < 0):
                output[i][j]=0
            else:
                output[i][j]=im[int(source_vec[0])][int(source_vec[1])]
            
    return output
