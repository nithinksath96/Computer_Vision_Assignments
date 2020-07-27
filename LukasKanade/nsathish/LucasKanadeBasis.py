import numpy as np
from scipy.interpolate import RectBivariateSpline
def Warp(It1 , rect, p0):
    

    
    height_It1 = np.arange(It1.shape[0])
        
    width_It1= np.arange(It1.shape[1])
    
    spline =  RectBivariateSpline(height_It1,width_It1,It1)
    
    x=  np.arange(rect[1][0] , rect[3][0]+1) 
    
    y = np.arange(rect[0][0] , rect[2][0]+1)  
    
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

def LucasKanadeBasis(It, It1, rect, bases ,p0=np.zeros(2)):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	bases: [n, m, k] where nxm is the size of the template.
	# Output:
	#	p: movement vector [dp_x, dp_y]

    # Put your implementation here
    del_p_norm = 10
    temp = p0 +0
    e= 0.1
    
    height = rect[2][0] -rect[0][0] +1
    
    width = rect[3][0] -rect[1][0] + 1
    
    

    
    x_cod_t , y_cod_t, spline_t = Warp(It,rect,temp)

    w=int(np.ceil(width))
    h=int(np.ceil(height))
    template_flatten = spline_t.ev(x_cod_t,y_cod_t).reshape(w*h,1)
    
    #template_vis =spline_t.ev(x_cod_t,y_cod_t).reshape(w,h)
#    plt.imshow(template_vis)
#    plt.pause(0.05)
#    plt.show()
#    #return

    
 
    bases_reshaped = np.reshape(bases , (bases.shape[0]*bases.shape[1],bases.shape[2]))
    bb= np.matmul(bases_reshaped , np.transpose(bases_reshaped))

   
    
    #print("HI")
    while(del_p_norm > e):
        
        print("DEl",del_p_norm)
    
        x_cod , y_cod , spline = Warp(It1,rect,temp)
        
        x_cod1 =x_cod + temp[0]
    
        y_cod1 =y_cod + temp[1]
        
        
        
        
        It1_W = spline.ev(x_cod1,y_cod1).reshape(w*h,1)
        It1W_x =spline.ev(x_cod1 , y_cod1 ,dx=1).reshape(w*h,1)
        It1W_y =spline.ev(x_cod1 , y_cod1 ,dy=1).reshape(w*h,1)
        
        
#      
        b = template_flatten - It1_W
        

        A = np.hstack((It1W_x, It1W_y))
        

        A = A[:bb.shape[0],:]
        b=  b[:bb.shape[0],:]
        
        A_1 = A - np.matmul(bb,A)
        b_1 = -np.matmul(bb,b) + b

        del_p,res,rank,s=np.linalg.lstsq(A_1,b_1 ,rcond=None)

        temp[0] = temp[0] + del_p[0]
        temp[1] = temp[1] + del_p[1]
        

        
        del_p_norm = np.linalg.norm(del_p) **2


    return temp


#data=np.load("../data/sylvseq.npy")
##
#frame1= data[:,:,0]
##
#frame2= data[:,:,1]
##
#rectangle = np.array([101,61,155,107]).reshape(4,1) 
##
##
#
#bases =np.load("../data/sylvbases.npy")
#p=LucasKanadeBasis(frame1 , frame2 , rectangle , bases) 
#print(p)