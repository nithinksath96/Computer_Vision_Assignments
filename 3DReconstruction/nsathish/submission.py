"""
Homework4.
Replace 'pass' by your implementation.
"""
import numpy as np
from helper import *
# Insert your package here

import cv2
from scipy import ndimage
from math import *
#import math
'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    #print(pts1.shape)
    N = int(pts1.shape[0])
    
    pts1_set = pts1[:N,:]

    pts2_set =pts2[:N,:]
    
    T = np.array([[1/M ,0,0],[0,1/M,0] ,[0,0,1]])
        
    padded_ones = np.ones((pts1_set.shape[0],1))
    
    pts1_hom = np.hstack((pts1_set , padded_ones))

    pts2_hom = np.hstack((pts2_set , padded_ones))
    
    pts1_norm = np.matmul(pts1_hom , T)
    
    pts2_norm = np.matmul(pts2_hom, T)

    for i in range(N):
        
        if(i==0):
            
            A = np.outer(pts2_norm[i,:] , pts1_norm[i,:]).flatten()
        else:
            A = np.vstack((A ,np.outer(pts2_norm[i,:] , pts1_norm[i,:]).flatten()))
 
    u,s,vt = np.linalg.svd(A)
    
    
    F= vt[-1,:].reshape(3,3)
        
    F_refined = refineF(F ,pts1_norm[:,:2] , pts2_norm[:,:2])
    
    F_unscale = np.matmul(np.matmul(np.transpose(T) , F_refined) , T)
    
    return F_unscale
    


    #pass


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''


    
def sevenpoint(pts1, pts2, M):
    # Replace pass by your implementation
    
    N= 7
    
    pts1_set = pts1
    
    pts2_set = pts2
    
    T = np.array([[1/M ,0,0],[0,1/M,0] ,[0,0,1]])

    padded_ones = np.ones((pts1_set.shape[0],1))
    
    pts1_hom = np.hstack((pts1_set , padded_ones))

    pts2_hom = np.hstack((pts2_set , padded_ones))
    
    pts1_norm = np.matmul(pts1_hom , T)
    
    pts2_norm = np.matmul(pts2_hom, T)

    
    for i in range(N):
        
        if(i==0):
            
            A = np.outer(pts2_norm[i,:] , pts1_norm[i,:]).flatten()
        
        else:
            
            A = np.vstack((A ,np.outer(pts2_norm[i,:] , pts1_norm[i,:]).flatten()))
 
    u,s,vt = np.linalg.svd(A)
    
    
    F1= vt[-1,:].reshape(3,3)
    
    F2 = vt[-2,:].reshape(3,3)
    
    fun = lambda a: np.linalg.det(a * F1 + (1 - a) * F2)
    
    a0 = fun(0)
    
    a1 = ((2*(fun(1) - fun(-1)))/3)  - ((fun(2) - fun(-2) / 12))
    
    a2 = 0.5*fun(1) + 0.5*fun(-1) -fun(0) 
    
    a3 = fun(1) - (a0 + a1 +a2)
    
    a_list = np.roots([a3,a2,a1,a0])
    
    a = a_list[np.isreal(a_list)]
    
    F_unscale_list = []
    
    for i in range(a.shape[0]):
        
        a_val = a[i].real
        
        F = a_val*F1 + (1-a_val)*F2
        
        #F_refined = refineF(F ,pts1_norm[:,:2] , pts2_norm[:,:2])
        
        F_unscale = np.matmul(np.matmul(np.transpose(T) , F) , T)
        
        F_unscale_list.append(F_unscale)
       
    return F_unscale_list 


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E = np.matmul(np.matmul(np.transpose(K2), F), K1)
    return E
    #pass


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation

    right_A = np.zeros((4,4))
    
    error=0
    
    P= np.zeros((pts1.shape[0],3))
    
    for i in range(pts1.shape[0]):
        right_A[0,:]  =pts1[i][0]*C1[2,:]  -C1[0,:]
        right_A[1,:] = pts1[i][1]*C1[2,:] - C1[1,:] 
        right_A[2,:]  =pts2[i][0]*C2[2,:]  -C2[0,:]
        right_A[3,:] = pts2[i][1]*C2[2,:] - C2[1,:]
        
        u,s,vt = np.linalg.svd(right_A)           
        w = vt[-1,:]
        
        w = w/w[3]
        
        P[i]= w[:3]
        
        x_camera_1 = np.matmul(C1,w)

        x_camera_2 = np.matmul(C2,w)

        x_camera_1 = x_camera_1 / x_camera_1[2]

        x_camera_2 = x_camera_2 / x_camera_2[2]
        
        error+= np.linalg.norm(x_camera_1[:2] - pts1[i]) + np.linalg.norm(x_camera_2[:2] - pts2[i])
    
    print("Error",error)

    return  [P ,error]      



'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
        
    window = 10
    
    im1 = np.copy(im1.astype(float))
    
    im2 = np.copy(im2.astype(float))

    point = np.array([[x1], [y1], [1]])
    
    l = np.dot(F, point)
    
    l = l / np.linalg.norm(l)

    pts_xy = np.empty((0, 2))

    pts_yx = np.empty((0, 2))

    if l[0] != 0:

        for y in range(window, im2.shape[0] - window):

            x = floor(-1.0 * (l[1] * y + l[2]) / l[0])

            if x >= window and x <= im2.shape[1] - window:

                pts_yx = np.append(pts_yx, np.array([x, y]).reshape(1, 2), axis=0)
    else:

        for x in range(window, im2.shape[1] - window):

            y = floor(-1.0 * (l[0] * x + l[2]) / l[1])

            if y >= window and y <= im2.shape[0] - window:

                pts_xy = np.append(pts_xy, np.array([x, y]).reshape(1, 2), axis=0)

    pts = pts_yx
    
    patch1 = im1[int(y1 - window + 1) : int(y1 + window), int(x1 - window + 1) : int(x1 + window), :]

    min_error = 1e12

    min_index = 0

    for i in range(pts.shape[0]):

        x2 = pts[i, 0]

        y2 = pts[i, 1]

        if sqrt((x1-x2)**2 + (y1-y2)**2) < 50:

            patch2 = im2[int(y2 - window + 1) : int(y2 + window), int(x2 - window + 1) : int(x2 + window), :]

            min_patch_size= min(patch1.shape[0],patch2.shape[0] ,patch1.shape[1],patch2.shape[1])

            patch1 = patch1[:min_patch_size,:min_patch_size]

            patch2 = patch2[:min_patch_size,:min_patch_size]

            error = patch1 - patch2

            error_filtered = np.sum(scipy.ndimage.gaussian_filter(error, sigma=1.0))

            if error_filtered < min_error:

                min_error = error_filtered

                min_index = i
    
    x2 = pts[min_index, 0]

    y2 = pts[min_index, 1]

    return x2, y2



'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M):
    # Replace pass by your implementation

    num_itr = 200
     
    thresh = 0.001
    
    max_inliers = 0
    
    np.random.seed(100)
    
    points1 = np.zeros((7, 2))
    
    points2 = np.zeros((7, 2))
    
    inliers_list = np.zeros((pts2.shape[0], 1), dtype=bool)
    
    for i in range(num_itr):
        
        print("Iteration", i)
        
        indexes = np.random.choice(pts2.shape[0], 7,replace=False)
        
        for id in range(indexes.shape[0]):
            points1[id] = pts1[indexes[id], 0:2]

            points2[id] = pts2[indexes[id], 0:2]
        
        
        F_list = sevenpoint(points1, points2, M)
        
        padded_ones = np.ones((pts1.shape[0],1))
            
        pts1_hom = np.hstack((pts1 , padded_ones))
#
        pts2_hom = np.hstack((pts2 , padded_ones))
        
        
        inliers_index = []
 
        for j in range(len(F_list)):
            
            inliers = 0

            pts1_inliers_list=np.zeros((1,2))
        
            pts2_inliers_list=np.zeros((1,2))

            for k in range(pts2_hom.shape[0]):

                error = abs(np.matmul(np.matmul(np.transpose(pts2_hom[k]),F_list[j]),pts1_hom[k]))

                if(error < thresh):
                    
                    inliers+=1
                    
                    pts1_inliers_list = np.vstack((pts1_inliers_list, pts1_hom[k][:2]))

                    pts2_inliers_list = np.vstack((pts2_inliers_list, pts2_hom[k][:2]))

                    inliers_index.append(k)
                                   
            print("Inliers",inliers)
                    
            if(inliers > max_inliers):
                
                max_inliers = inliers
                
                bestF = F_list[j]
                
                best_inliers_pts1 = pts1_inliers_list[1:]
 
                best_inliers_pts2 = pts2_inliers_list[1:]
                
                best_inlier_index = np.array(inliers_index)
    
        print("Max inliers", max_inliers)  
    
    print("Shape",best_inliers_pts1.shape)
    
    refined_F = eightpoint(best_inliers_pts1,best_inliers_pts2,M)
    
    print("F",refined_F)
    
    for index in best_inlier_index:
    
        inliers_list[index] = 1
    
    return refined_F , inliers_list

'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    print("r",r)
    
    theta = np.linalg.norm(r)
    
    u=r/theta
    
    ux = np.array([[0,-r[2][0],r[1][0]],
                   [r[2][0],0,-r[0][0]],
                    [-r[2][0],r[0][0],0]])
    if(theta==0):
    
        return np.identity(3)
    
    R = np.identity(3)*np.cos(theta) + (1-np.cos(theta))*(np.matmul(u,np.transpose(u))) + ux*np.sin(theta)

    return R


'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    
    
    # Replace pass by your implementation
    epsilon = 1e-16
    
    print("R",R)
    
    print(R.shape)
    
    print("traceR",np.trace(R))
    
    theta = acos((np.trace(R) - 1) / 2.0)
    
    r = np.zeros((3, 1))

    if abs(theta) > epsilon:
    
        norm_axis = 1.0 / (2*sin(theta)) * np.array([[R[2, 1] - R[1, 2]],
                                                     [R[0, 2] - R[2, 0]],
                                                     [R[1, 0] - R[0, 1]]])
        r = theta * norm_axis
    
    return r

            
    #pass

'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    
    # Replace pass by your implementation

    t = x[-3:].reshape(3, 1)

    rod_v = x[-6:-3].reshape(3, 1)

    P = x[0:-6].reshape(-1, 3)
        
    R = rodrigues(rod_v);
    
    M2 = np.append(R, t, axis=1)
    
    C1 = np.dot(K1, M1)
    
    C2 = np.dot(K2, M2)

    P_homo = np.append(P, np.ones((P.shape[0], 1)), axis=1).transpose()

    p1_reprojected = np.dot(C1, P_homo)
    
    p2_reprojected = np.dot(C2, P_homo)

    p1_normalized = np.zeros((2, P_homo.shape[1]))
    
    p2_normalized = np.zeros((2, P_homo.shape[1]))

    p1_normalized[0, :] = p1_reprojected[0, :] / p1_reprojected[2, :]
    
    p1_normalized[1, :] = p1_reprojected[1, :] / p1_reprojected[2, :]
    
    p2_normalized[0, :] = p2_reprojected[0, :] / p2_reprojected[2, :]
    
    p2_normalized[1, :] = p2_reprojected[1, :] /p2_reprojected[2, :]
    
    p1_normalized = p1_normalized.transpose()
    
    p2_normalized = p2_normalized.transpose()

    error1 = (p1 - p1_normalized).reshape(-1)
    
    error2 = (p2 - p2_normalized).reshape(-1)

    residuals = np.append(error1, error2, axis=0)
    
    print("residuals",residuals)

    return residuals


'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    
    # Replace pass by your implementation
    #print("BA",M2_init)
    residual = lambda x: rodriguesResidual(K1, M1, p1, K2, p2, x)
    
    R2_init = M2_init[:, 0:3]
    
    t2_init = M2_init[:, 3]
    
    r2_init = invRodrigues(R2_init).reshape(-1)
    
    x_init = np.zeros(3 * P_init.shape[0] + 6)
  
    x_init[-3:] = t2_init
    
    x_init[-6:-3] = r2_init
    
    x_init[0:-6] = P_init.flatten()
    
    x_final,cost = scipy.optimize.leastsq(residual ,x_init)
    
    print("output of optimizer", x_final)
    
    t = x_final[-3:].reshape(3,1)
    
    r = x_final[-6:-3].reshape(3, 1)
    
    P = x_final[0:-6].reshape(-1, 3)

    R = rodrigues(r)
    
    M2 = np.hstack((R,t))

    
    return M2, P


