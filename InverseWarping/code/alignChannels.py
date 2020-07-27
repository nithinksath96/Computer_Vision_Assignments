import numpy as np
import scipy.ndimage
import cv2 as cv
from PIL import Image
def SSD(vec1,vec2):
    x=vec1-vec2
    return np.sum(np.multiply(x,x))



def alignChannels(red, green, blue):
    """Given 3 images corresponding to different channels of a color image,
    compute the best aligned result with minimum abberations

    Args:
      red, green, blue - each is a HxW matrix corresponding to an HxW image

    Returns:
      rgb_output - HxWx3 color image output, aligned as desired"""

    red_ssd={}
    blue_ssd={}

    blue_cropped= blue[250:650, 250:650]
    red_cropped=  red[250:650, 250:650]

    for i in range(220,280):

        # red_ver = np.roll(red,i,axis=1)
        # green_ver = np.roll(green,i,axis=1)

        for j in range(220,280):
            #SSD(red,blue)
            #SSD(green,blue)

            # red_hor=np.roll(red_ver,red.shape[1]*j)
            # green_hor=np.roll(green_ver,green.shape[1]*j)
            #print(red_cropped.shape)
            #print(green[i:i+300, j:j+300].shape)

            red_ssd[i,j]=SSD(red_cropped,green[i:i+400, j:j+400])
            blue_ssd[i,j]=SSD(blue_cropped,green[i:i+400, j:j+400])

            #NCC(red,blue)
            #NCC(green,blue)

    r_row,r_col=min(red_ssd, key=red_ssd.get)
    b_row,b_col=min(blue_ssd, key=blue_ssd.get)


    #print(red_ssd[r_row,r_col])
    #print(green_ssd[g_row,g_col])

    print("Val Red",min(red_ssd.values()))  
    print("Val green",min(blue_ssd.values()))

    print("key red",r_row,r_col)
    print("key blue",b_row,b_col)



    temp=np.roll(red,(r_col-250),axis=1)
    red_f=np.roll(temp,(r_row-250),axis=0)


    temp=np.roll(blue,(b_col-250),axis=1)
    blue_f=np.roll(temp,(b_row-250),axis=0)

    #red_n=Image.fromarray(red_f)
    #blue_n=Image.fromarray(blue_f)
    #green_n=Image.fromarray(green)

    #rgb=Image.merge("RGB",(red_n,green_n,blue_n))
    #rgb.show()

    #red_f= scipy.ndimage.shift(red,(r_row-250,r_col-250))
    #blue_f=scipy.ndimage.shift(blue,(b_row-250,b_col-250))


    rgb_output=np.dstack((red_f,green,blue_f))

    #rgb_original=np.dstack((red,green,blue))
    #cv.imshow("Output",rgb_output)
    #cv.imshow("Original",rgb_original)
    #cv.waitKey(0)
    return rgb_output
