import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import cv2
from LucasKanade import LucasKanade
# write your script here, we recommend the above libraries for making your animation


    

if __name__ == "__main__":
    
    data=np.load("../data/carseq.npy")
    
    rect = np.array([59,116,145,151]).reshape(4,1)
    
    rect2 =rect+0
    
    initial_template = np.array([59,116,145,151]).reshape(4,1)

    p = np.zeros(2)
    
    fig=plt.figure()
    ax1 = fig.add_subplot(111)

    #print(ax1)
    width = rect[2][0] - rect[0][0] 
    height = rect[3][0] -rect[1][0]
    frames =[1,100,200,300,400]
    I0 = data[:,:,0]
    rect_list =rect.T
    
    #print("Rect",rect)
    correction_e =1
    
    for i in range(data.shape[2]-1):
        
        It=data[:,:,i]
        It1=data[:,:,i+1]
        
        
        #print(It.shape)
        #print(np.transpose(It).shape)
                
        rect =rect.astype(float)
        rect2 =rect2.astype(float)


        
        p1 = LucasKanade(It, It1, rect , p)
        
        #rect2[0][0]= float(rect2[0][0] + p1[1])
            
            #print("Type",rect.dtype)
            #print("-----------------------------------------------",rect[0][0])
            #break
        #rect2[2][0]= float(rect2[2][0] + p1[1])
        #rect2[1][0]= float(rect2[1][0] + p1[0])
        #rect2[3][0] = float(rect2[3][0] + p1[0])
        
        
        
        #print(p1)
        #print("Here")
        p1_update  = np.array((rect[1] + p1[0] - initial_template[1] , rect[0] + p1[1] - initial_template[0])).reshape(2)
        print("Update",p1_update)
        #break
        p_correction = LucasKanade(I0, It1, initial_template ,p1_update)
        #print(p_correction)
        
        
        #print("Naive",p1)
        #print("Correction",p_correction)
        correction_diff = p_correction - p1_update 
        #print("DIff",np.linalg.norm(correction_diff))
        #print("Rect",rect)
        if(np.linalg.norm(correction_diff) <= correction_e):
            
            print("-------------------------First case")
            rect[0][0]= float(rect[0][0] + p1[1])
            rect[2][0]= float(rect[2][0] + p1[1])
            rect[1][0]= float(rect[1][0] + p1[0])
            rect[3][0] = float(rect[3][0] + p1[0])
            
        else:
            print("------------------------------------Second case")
            rect[0][0]= float(initial_template[0][0] + p_correction[1])
            rect[2][0]= float(initial_template[2][0] + p_correction[1])
            rect[1][0]= float(initial_template[1][0] + p_correction[0])
            rect[3][0] = float(initial_template[3][0] + p_correction[0])
            
    
  
#        print("Did not enter")
        
       
   
        
        
        rect_list = np.vstack((rect_list , rect.T))
#        print("Parameter",p1)
#        
#        print("Rect",rect)
    

        It1=np.float32(It1)

        #if(i==100):
        #if(i==300): 
        It2 = cv2.cvtColor(It1,cv2.COLOR_GRAY2RGB)
        ax1.add_patch(patches.Rectangle((rect[0][0] , rect[1][0]) , width, height , fill=False , color = 'red'))
        
#        if(i==1):
#            cv2.rectangle(It2,(rect[0][0],rect[1][0]), (rect[2][0], rect[3][0]) , (0,0,255))
#            cv2.imshow("Image", It2)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
#        
        #ax1.add_patch(patches.Rectangle((rect2[0][0] , rect2[1][0]) , width, height , fill=False , color = 'green'))
        plt.show(block=False)
        plt.imshow(It2 ,cmap='gray')
        plt.pause(0.05)
        #plt.pause(1)
        #break
        #input("Press [enter] to continue.")

        ax1.clear()
        
    np.save("carseqrects-wcrt.npy",rect_list)