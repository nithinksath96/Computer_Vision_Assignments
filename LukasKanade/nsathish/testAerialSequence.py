import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches


from LucasKanadeAffine import LucasKanadeAffine
from SubtractDominantMotion import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation
if __name__ == "__main__":
    
    data=np.load("../data/aerialseq.npy")
    
    frames = [30,60,90,120]
    
    mask_list =np.zeros((240,320))
    
    for i in range(data.shape[2]-1):
        
        
        It=data[:,:,i]
        
        It1=data[:,:,i+1]
        
        mask = SubtractDominantMotion(It , It1)
        
        if(i in frames):
            mask_list = np.dstack((mask_list,mask))
        
    
        
        It1_copy = It1.copy()
        
        It1_rgb = np.dstack((It1_copy,It1_copy,It1_copy)) *255.0
        
        It1_rgb[:,:,2] += mask *150.0
        
        It1_rgb =np.clip(It1_rgb ,0, 255)
        
        It1_rgb =It1_rgb.astype(np.uint8)
        
        plt.imshow(It1_rgb)
        plt.show()
    np.save("aerialseqrects.npy",mask_list)   
    #print(mask_list.shape)