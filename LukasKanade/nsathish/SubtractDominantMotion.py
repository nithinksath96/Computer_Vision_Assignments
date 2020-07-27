import numpy as np

from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine
from scipy.ndimage import affine_transform
from scipy.ndimage import binary_dilation

def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
    # put your implementation here
    
    mask = np.ones(image1.shape, dtype=bool)
    
    M = LucasKanadeAffine(image1 , image2)
    #M = InverseCompositionAffine(image1, image2)
    
    padded = np.array([0,0,1])
    
    M_1 = np.vstack((M,padded))
    
    warped_image = affine_transform(image1, np.linalg.inv(M_1))
    
    m  = np.zeros((warped_image.shape))
    
    m[:-30,40:-40] =1.0
    
    abs_I = abs(image2 - warped_image) * m
    

    abs_I1 = np.where(abs_I > 0.2 , 1,0)
    
    mask = binary_dilation(abs_I1, structure=np.ones((10,10)))
    
    #print(mask.shape)
    return mask

