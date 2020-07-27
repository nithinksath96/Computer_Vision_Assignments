'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np


from submission import *
import cv2 
from helper import *
import matplotlib.image as mpimg
from findM2 import *



from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

def Bundle_Adjustment():
    
    data = np.load('../data/some_corresp_noisy.npz')
    
    pts1 = data['pts1']
    
    pts2 = data['pts2']
    
    intrinsics = np.load('../data/intrinsics.npz')
    
    image1 = mpimg.imread('../data/im1.png')    
    
    image2 =mpimg.imread("../data/im1.png")
    
    M = max(image1.shape[:2])
    
    F_ransac , inliers = ransacF(pts1,pts2,M)
    
    #print(inliers.shape)
    
    pts1_in = np.empty((0, 2))
    
    pts2_in = np.empty((0, 2))

    for i in range(inliers.shape[0]):
    
        if inliers[i] == True:
        
            pts1_in = np.append(pts1_in, pts1[i].reshape(1, 2), axis=0)
            
            pts2_in = np.append(pts2_in, pts2[i].reshape(1, 2), axis=0)
    
    #displayEpipolarF(image1,image2,F_ransac)
    
    #pass
    
    F = eightpoint(pts1,pts2,M)
    
    #displayEpipolarF(image1,image2,F)
    #displayEpipolarF(image1,image2,F_ransac)    
    
    K1 = intrinsics['K1']    
    
    K2= intrinsics['K2']    
    
    E = essentialMatrix(F_ransac,K1,K2)    
    
    camera_mat = np.identity(3)    
    
    zero_padding = np.zeros((3,1))    
    
    M1 = np.hstack((camera_mat,zero_padding))    
    
    M2 = camera2(E)    
    
    C2 = np.zeros((3,4,4))    
    
    C1 = np.matmul(K1 , M1)    
    
    for i in range(M2.shape[2]):        
    
        C2[:,:,i] = np.matmul(K2 , M2[:,:,i])    
    
    A = np.zeros((4,4,4))
    
    neg_z = np.zeros(4)    

    for i in range(A.shape[2]):
        
        for j in range(pts1_in.shape[0]):
        
            A[0,:,i]  =pts1_in[j][0]*C1[2,:]  -C1[0,:]
            
            A[1,:,i] = pts1_in[j][1]*C1[2,:] - C1[1,:]
            
            A[2,:,i]  =pts2_in[j][0]*C2[2,:,i]  -C2[0,:,i]
            
            A[3,:,i] = pts2_in[j][1]*C2[2,:,i] - C2[1,:,i]
            
            u,s,vt = np.linalg.svd(A[:,:,i])           
            
            w = vt[-1,:]
            
            w = w/w[3]
            
            if(w[2]< 0):
            
                neg_z[i]+=1
                
    rightC2 = C2[:,:,np.argmin(neg_z)]
    
    P,err_before=triangulate(C1, pts1_in,rightC2,pts2_in)
    
    M2_ba, P_ba = bundleAdjustment(K1, M1, pts1_in, K2, M2[:,:,np.argmin(neg_z)], pts2_in, P)
    
    C2_ba = np.dot(K2, M2_ba)

    P,err=triangulate(C1, pts1_in,C2_ba,pts2_in)
    
    #print("after bundle adjustment",err)
    
    #print("Before bundle adjustment",err_before)
    
    #bundleAdjustment(K1, M1, pts1_in, K2, M2[:,:,np.argmin(neg_z)], pts2_in, P)

    fig = pyplot.figure()
    
    ax = Axes3D(fig)

    ax.scatter(P[:,0],P[:,1],P[:,2])
    
    pyplot.show()
    
def visualization():
    data = np.load('../data/some_corresp.npz')

    pts1 = data['pts1']
    
    pts2 = data['pts2']
     
    intrinsics = np.load('../data/intrinsics.npz')
    
    image1 = mpimg.imread('../data/im1.png')    
    
    image2 =mpimg.imread("../data/im1.png")
    
    M = max(image1.shape[:2])
        
    F = eightpoint(pts1,pts2,M)    
    
    K1 = intrinsics['K1']    
    
    K2= intrinsics['K2']    
    
    E = essentialMatrix(F,K1,K2)    
    
    camera_mat = np.identity(3)    
    
    zero_padding = np.zeros((3,1))    
    
    M1 = np.hstack((camera_mat,zero_padding))    
    
    M2 = camera2(E)    
    
    C2 = np.zeros((3,4,4))    
    
    C1 = np.matmul(K1 , M1)    
    
    for i in range(M2.shape[2]):        
    
        C2[:,:,i] = np.matmul(K2 , M2[:,:,i])    
    
    #np.savez('q4_1.npz', F=F)
    
    #epipolarMatchGUI(image1,image2,F)
        
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
    
    img1_points = np.load('../data/templeCoords.npz')
    
    x1 = img1_points['x1']
    
    y1 = img1_points['y1']
    
    points2 = np.zeros((x1.shape[0],2))
    
    print(x1.shape[0])
    
    for i in range(x1.shape[0]):
    
        print("I",i)
        
        points2[i,0],points2[i,1] = epipolarCorrespondence(image1,image2,F,x1[i][0],y1[i][0]) 
    
    points1= np.hstack((x1,y1))
     
    P, err = triangulate(C1,points1,rightC2,points2)
    
    #np.savez('q4_2.npz',F=F, M1=M1, M2=M2[:,:,np.argmin(neg_z)], C1=C1, C2=rightC2)

    fig = pyplot.figure()
    
    ax = Axes3D(fig)

    ax.scatter(P[:,0],P[:,1],P[:,2])
    
    pyplot.show()

if __name__ == '__main__':
    visualization()
    Bundle_Adjustment()
    
    