import numpy as np
from scipy.interpolate import RectBivariateSpline


#### DELETE THIS
import cv2
from matplotlib import animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def Warp(It1 , rect, p0):
    

    
    height_It1 = np.arange(It1.shape[0])
        
    width_It1= np.arange(It1.shape[1])
    
    spline =  RectBivariateSpline(height_It1,width_It1,It1)
    
    x=  np.arange(rect[1][0] , rect[3][0]) 
    
    y = np.arange(rect[0][0] , rect[2][0])  
    
    #print("X",x)
    #print("Y",y)
    
    xx,yy = np.meshgrid(x,y)
    

    
    
         
    x_cod = xx.flatten('F')
    
    y_cod = yy.flatten('F')
    
    #print(x_cod)
    #print(y_cod)
    
    #print("A",xx)
    #print("B",yy)
    
    

    

    #It1_W = spline.ev(x_cod,y_cod).reshape(height,width)
    #It1_W = spline.ev(x_cod , y_cod).reshape(width , height)
    
    #print(It1_W[1])
    
    #It1_W = spline.ev(x_cod , y_cod)
    return x_cod , y_cod , spline 


    
    
def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]
	
    # Put your implementation here
    del_p_norm = 10
    temp = p0 +0
    e= 0.1
    
    height = rect[2][0] -rect[0][0] 
    
    width = rect[3][0] -rect[1][0]
    
    

    
    x_cod_t , y_cod_t, spline_t = Warp(It,rect,temp)

    w=int(np.ceil(width))
    h=int(np.ceil(height))
    template_flatten = spline_t.ev(x_cod_t,y_cod_t).reshape(w*h,1)
    
    #template_vis =spline_t.ev(x_cod_t,y_cod_t).reshape(w,h)
#    plt.imshow(template_vis)
#    plt.pause(0.05)
#    plt.show()
#    #return

    
 
    

   
    
    #print("HI")
    while(del_p_norm > e):
        
    
        x_cod , y_cod , spline = Warp(It1,rect,temp)
        
        x_cod1 =x_cod + temp[0]
    
        y_cod1 =y_cod + temp[1]

        It1_W = spline.ev(x_cod1,y_cod1).reshape(w*h,1)
        It1W_x =spline.ev(x_cod1 , y_cod1 ,dx=1).reshape(w*h,1)
        It1W_y =spline.ev(x_cod1 , y_cod1 ,dy=1).reshape(w*h,1)
        
        
#      
        b = template_flatten - It1_W
        

        A = np.hstack((It1W_x, It1W_y))

   

        del_p,res,rank,s=np.linalg.lstsq(A,b ,rcond=None)

        temp[0] = temp[0] + del_p[0]
        temp[1] = temp[1] + del_p[1]
        

        
        del_p_norm = np.linalg.norm(del_p) **2


    return temp


#data=np.load("../data/carseq.npy")
#
#frame1= data[:,:,0]
#
#frame2= data[:,:,1]
#
#rectangle = np.array([59,116,145,151]).reshape(4,1) 
#
#
#p=LucasKanade(frame1 , frame2 , rectangle)










#print(p)
#cv2.imshow("FRAME1", frame1)
#cv2.imshow("Frame2",frame2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()