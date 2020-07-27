import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

##Delete
import cv2
# write your script here, we recommend the above libraries for making your animation5

from LucasKanade import LucasKanade

if __name__ == "__main__":
    
    data=np.load("../data/carseq.npy")
    
    rect = np.array([59,116,145,151]).reshape(4,1)

    p = np.zeros(2)
    
    fig=plt.figure()
    ax1 = fig.add_subplot(111)
    #print(ax1)
    width = rect[2][0] - rect[0][0] 
    height = rect[3][0] -rect[1][0]
    frames =[1,100,200,300,400]
    rect_list =rect.T
    for i in range(data.shape[2]-1):
        It=data[:,:,i]
        
        It1=data[:,:,i+1]
        
        
        #print(It.shape)
        #print(np.transpose(It).shape)
        
        
        p1 = LucasKanade(It, It1, rect , p)
        
        
        rect=rect.astype(float)
        rect[0][0]= float(rect[0][0] + p1[1])
        #print("Type",rect.dtype)
        #print("-----------------------------------------------",rect[0][0])
        #break
        rect[2][0]= float(rect[2][0] + p1[1])
        rect[1][0]= float(rect[1][0] + p1[0])
        rect[3][0] = float(rect[3][0] + p1[0])
        rect_list = np.vstack((rect_list , rect.T))
        print("Parameter",p1)
        
        print("Rect",rect)
    

        It1=np.float32(It1)

        #if(i==100):
        #if(i==300): 
        It2 = cv2.cvtColor(It1,cv2.COLOR_GRAY2RGB)
        ax1.add_patch(patches.Rectangle((rect[0][0] , rect[1][0]) , width, height , fill=False , color = 'red'))
        
        plt.imshow(It2 ,cmap='gray')
        plt.pause(0.05)
        ax1.clear()
            #break
            #cv2.imwrite(filename,It2)
        #break
        
        
        #cv2.rectangle(It2,(rect[0][0],rect[1][0]), (rect[2][0], rect[3][0]) , (0,0,255))
        
        #cv2.imshow("Image", It2)
        #cv2.waitKey(200)
        #cv2.destroyAllWindows()
        

        
        
    np.save("carseqrects.npy",rect_list)     