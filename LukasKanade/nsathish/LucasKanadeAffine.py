import numpy as np
from scipy.interpolate import RectBivariateSpline


from scipy.ndimage import affine_transform
import cv2
import matplotlib.pyplot as plt




def LucasKanadeAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    
    del_p_norm = 10
    temp = M +0
    e= 0.1
    
#    It1 = It1[:-30,40:-40]
#    It = It[:-30,40:-40]

    template_flatten = np.reshape(It , (It.shape[0]*It.shape[1],1))
    

        
    #print("HI")
    while(del_p_norm > e):
        
        print("DEl",del_p_norm)

        
       

        height_It1 = np.arange(It1.shape[0])
        
        width_It1= np.arange(It1.shape[1])
    
        spline =  RectBivariateSpline(height_It1,width_It1,It1)
        
        h=It.shape[0]
        
        w=It.shape[1]
        

        
        pad_M = np.array([0,0,1])
#        
        M_1 =np.vstack((temp,pad_M))
        
        mask = np.ones((It1.shape[0], It1.shape[1]))
        
        mask_warped=affine_transform(mask ,np.linalg.inv(M_1))
        
        warped_source = np.multiply(mask_warped, It1).reshape(w*h,1)
        
        dx,dy = np.gradient(It1)
        
        
        dx_warp =affine_transform(dx , np.linalg.inv(M_1)).reshape(w*h,1)
        
        dy_warp = affine_transform(dy , np.linalg.inv(M_1)).reshape(w*h,1)
        

        gradients = np.hstack((dx_warp , dy_warp))
        
   
        
        
        Jacobian = np.zeros((w*h,2,6))
        
        x1 = np.arange(h)
       
        y1 = np.arange(w)
                
        xx1,yy1 = np.meshgrid(x1,y1)
        
        x_cod_it1 = xx1.flatten('F')
    
        y_cod_it1= yy1.flatten('F')


        Jacobian[:,0,0] = x_cod_it1
        
        Jacobian[:,0,1] = y_cod_it1
        
        Jacobian[:,0,2] = 1
        
        Jacobian[:,1,3] = x_cod_it1
        
        Jacobian[:,1,4] = y_cod_it1
        
        Jacobian[:,1,5] = 1
        
        b = template_flatten  - warped_source
        
        A=np.zeros((Jacobian.shape[0],6))
        
        for j in range(Jacobian.shape[0]):
            
            A[j] =  np.matmul(gradients[j,:],Jacobian[j,:,:])
            
        del_p,res,rank,s=np.linalg.lstsq(A,b ,rcond=None)
        
        temp[0][0]+= del_p[0]
        
        temp[0][1]+=del_p[1]
        
        temp[0][2]+=del_p[2]
        
        temp[1][0]+= del_p[3]
        
        temp[1][1]+=del_p[4]
        
        temp[1][2]+=del_p[5]
        

        del_p_norm = np.linalg.norm(del_p) **2
    
    
    return temp
        
            
        
        
        
        
        #print(warped_source.shape)
        
#        Jacobian = np.zeros((xy_warped_cod.shape[0],2,6))
#        h1=It1.shape[0]
#        
#        w1=It1.shape[1]
#        
#        x1 = np.arange(h1)
#        
#        y1 = np.arange(w1)
#                
#        xx1,yy1 = np.meshgrid(x1,y1)
#        
#        x_cod_it1 = xx1.flatten('F')
#    
#        y_cod_it1= yy1.flatten('F')
#
#
#        Jacobian[:,0,0] = x_cod_it1
#        
#        Jacobian[:,0,2] = y_cod_it1
#        
#        Jacobian[:,0,4] = 1
#        
#        Jacobian[:,1,1] = x_cod_it1
#        
#        Jacobian[:,1,3] = y_cod_it1
#        
#        Jacobian[:,1,5] = 1
#        
#        J = np.reshape(Jacobian , (Jacobian.shape[0]*Jacobian.shape[2],Jacobian.shape[1]))
#        
#        b= template_flatten - It1_W
#        
#        
#        del_p,res,rank,s=np.linalg.lstsq(A,b ,rcond=None)
#
#        temp[0] = temp[0] + del_p[0]
#        temp[1] = temp[1] + del_p[1]
        
        
        #print(J.shape)
        

        
        
        
        
        
        
        
        
#        
#        cv2.imshow("derivativex", It1W_x)
#        cv2.imshow("derivativey", It1W_y)
#        cv2.imshow("derivative", It1_W)
##        
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
#        return 
#        











#data=np.load("../data/aerialseq.npy")
##
#frame1= data[:,:,0]
##
#frame2= data[:,:,1]
##
##rectangle = np.array([59,116,145,151]).reshape(4,1) 
##
##
#M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
##affine_op = affine_transform(frame1,M)
###print(affine_op[-1,:])
##plt.imshow(affine_op)
##plt.show()
#p=LucasKanadeAffine(frame1 , frame2)