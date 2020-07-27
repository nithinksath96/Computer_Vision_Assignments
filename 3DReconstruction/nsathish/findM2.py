import numpy as np


from submission import *
import cv2 
from helper import *
import matplotlib.image as mpimg



from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
def find_points(pts1,pts2):
    
    F_ref = np.array([[ 9.78833288e-10 ,-1.32135929e-07 , 1.12585666e-03],
                      [-5.73843315e-08  ,2.96800276e-09 ,-1.17611996e-05],
                      [-1.08269003e-03  ,3.04846703e-05 ,-4.47032655e-03]])
    
    padded_ones = np.ones((pts1.shape[0],1))
    
    pts1_hom = np.hstack((pts1 , padded_ones))

    pts2_hom = np.hstack((pts2 , padded_ones))
    
    error = np.array([0])
      
    for i in range(pts2_hom.shape[0]):
        
        error = abs(np.append(error, np.matmul(np.matmul(pts2_hom[i],F_ref),pts1_hom[i])))
        
    error = error[1:]
    
    idx = np.argsort(error)[:7]
    
    return pts1[idx] ,pts2[idx]

def test_M2_solution(pts1, pts2, intrinsics , M):
    '''
    Estimate all possible M2 and return the correct M2 and 3D points P
    :param pred_pts1:
    :param pred_pts2:
    :param intrinsics:
    :return: M2, the extrinsics of camera 2
             C2, the 3x4 camera matrix
             P, 3D points after triangulation (Nx3)
    '''
      

    image1 = mpimg.imread('../data/im1.png')
    
    image2 =mpimg.imread("../data/im1.png")

    F = eightpoint(pts1,pts2,M)
    
    #print(F)

    #np.savez('q2_1.npz', F=F, M=M)

    #displayEpipolarF(image1,image2, F)
    
    selected_points1 , selected_points2= find_points(pts1,pts2)
    
    F_list = sevenpoint(selected_points1,selected_points2,M)
    
    #print(F_list)
    
    #np.savez('q2_2.npz',F=F_list[0], M=M ,pts1=selected_points1, pts2=selected_points2)
    
    #displayEpipolarF(image1,image2, F_list[2])
##        
    K1 = intrinsics['K1']
    
    K2= intrinsics['K2']
    
    E = essentialMatrix(F,K1,K2)
    
    #print(E)
#        
    camera_mat = np.identity(3)
    
    zero_padding = np.zeros((3,1))
    
    M1 = np.hstack((camera_mat,zero_padding))
    
    M2 = camera2(E)
    
    C2= np.zeros((3,4,4))
    
    C1 = np.matmul(K1 , M1)
        
    for i in range(M2.shape[2]):
        
        C2[:,:,i] = np.matmul(K2 , M2[:,:,i])
    
    A = np.zeros((4,4,4))
    
    neg_z = np.zeros(4)    

    for i in range(A.shape[2]):
        
        for j in range(pts1.shape[0]):
            
            A[0,:,i]  =pts1[j][0]*C1[2,:]  -C1[0,:]
            
            A[1,:,i] = pts1[j][1]*C1[2,:] - C1[1,:]
            
            A[2,:,i]  =pts2[j][0]*C2[2,:,i]  -C2[0,:,i]
            
            A[3,:,i] = pts2[j][1]*C2[2,:,i] - C2[1,:,i]
            
            u,s,vt = np.linalg.svd(A[:,:,i])           
            
            w = vt[-1,:]
            
            w = w/w[3]
            
            if(w[2]< 0):
            
                neg_z[i]+=1

    rightC2 = C2[:,:,np.argmin(neg_z)]
    
    P, err = triangulate(C1,pts1,rightC2,pts2)
        
    M2 = M2[:,:,np.argmin(neg_z)]
    
    C2 = rightC2
    
    #np.savez('q3_3.npz', M2=M2, C2=C2, P=P)
#    
    return M2, C2, P


if __name__ == '__main__':
    data = np.load('../data/some_corresp.npz')
    
    pts1 = data['pts1']
   
    pts2 = data['pts2']
    
    intrinsics = np.load('../data/intrinsics.npz')
    
    image1 = mpimg.imread('../data/im1.png')
    
    image2 =mpimg.imread("../data/im1.png")

    M = max(image1.shape[:2])

    M2, C2, P = test_M2_solution(pts1, pts2, intrinsics, M)

