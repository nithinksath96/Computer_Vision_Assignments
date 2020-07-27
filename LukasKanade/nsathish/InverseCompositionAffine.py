import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform

def InverseCompositionAffine(It, It1):
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
      
    It1 = It1[:-30,40:-40]
    
    It = It[:-30,40:-40]

    template_flatten = np.reshape(It , (It.shape[0]*It.shape[1],1))
    
    dx,dy = np.gradient(It)
    
    dx = dx.flatten()
    
    dy = dy.flatten()
    
    h=It.shape[0]
        
    w=It.shape[1]
        
    pad_M = np.array([0,0,1])
#        
    M_1 =np.vstack((temp,pad_M))

        
    gradients = np.array([dx,dy]).T
        
    print("grad", gradients.shape)
    
    Jacobian = np.zeros((w*h,2,6))
        
    x1 = np.arange(h)
       
    y1 = np.arange(w)
                
    xx1,yy1 = np.meshgrid(x1,y1)
        
    x_cod_it = xx1.flatten('F')
    
    y_cod_it= yy1.flatten('F')


    Jacobian[:,0,0] = x_cod_it
        
    Jacobian[:,0,1] = y_cod_it
        
    Jacobian[:,0,2] = 1
        
    Jacobian[:,1,3] = x_cod_it
        
    Jacobian[:,1,4] = y_cod_it
        
    Jacobian[:,1,5] = 1
     
    A=np.zeros((Jacobian.shape[0],6))
    

        
    for j in range(Jacobian.shape[0]):
            
        A[j] =  np.matmul(gradients[j,:],Jacobian[j,:,:])
         
    A_1 = np.matmul(np.linalg.inv(np.matmul(A.T,A)), A.T)
        
     
        
        
    #print("HI")
    i = 0
    while(del_p_norm > e):
        


        mask = np.ones((It1.shape[0], It1.shape[1]))
        
        pad_M = np.array([0,0,1])
        
        M_1 =np.vstack((temp,pad_M))

        template_flatten = affine_transform(It, np.linalg.inv(M_1))
        
        template_flatten = template_flatten.flatten()

        mask_warped=affine_transform(mask ,np.linalg.inv(M_1))

        warped_source = np.multiply(mask_warped, It1).reshape(w*h,1)

        template_flatten = np.reshape(template_flatten , (template_flatten.shape[0],1))

        
        b =  warped_source - template_flatten
        
        del_p,res,rank,s=np.linalg.lstsq(A,b ,rcond=None)
        
        temp[0][0]+= del_p[0]
        
        temp[0][1]+=del_p[1]
        
        temp[0][2]+=del_p[2]
        
        temp[1][0]+= del_p[3]
        
        temp[1][1]+=del_p[4]
        
        temp[1][2]+=del_p[5]
        
#
#        del_p = np.matmul(A_1, b)
#
#
#        
#        del_p[0]+= 1 + del_p[0]
#        
#        del_p[4]+=1 + del_p[4]
#        
#        del_p = np.reshape(del_p , (2,3))
#        
#        del_p= np.vstack((del_p ,np.array([0,0,1])))
#
#        temp = np.matmul(temp, np.linalg.inv(del_p))

        
        del_p_norm = np.linalg.norm(del_p) **2

        i += 1
    
    
    return temp

