import numpy as np
import cv2
from BRIEF import briefLite, briefMatch
from numpy.linalg import inv
import scipy

def computeH(p1, p2):
    '''
    INPUTS
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                coordinates between two images
    OUTPUTS
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
                equation
    '''
    
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    # TO DO ...
    
    A=np.zeros((2*p1.shape[1],9))
    
    A[1::2,0] =p1[0,:]
    A[1::2,1] =p1[1,:]
    A[1::2,2] =1
    
    A[0::2,3] =-p1[0,:]
    A[0::2,4] =-p1[1,:]
    A[0::2,5] =-1
    

    
    A[0::2,6] = p1[0,:]*p2[1,:]
    A[1::2,6] = -p1[0,:]*p2[0,:]
    
    A[0::2,7] = p1[1,:]*p2[1,:]
    A[1::2,7] = -p1[1,:]*p2[0,:]
    
    A[0::2,8] = p2[1,:]
    A[1::2,8] =-p2[0,:]
    
   
    

    
    u, s, vh = np.linalg.svd(A)
    
    #print(vh)
    
    
    #print("P2",p2)
    
    H2to1= vh[8,:].reshape(3,3)
    #H2to1
    #print(np.array([p1[0],p1[1]]))
    #H2to1, status = cv2.findHomography(p1, p2)
    H2to1= H2to1/H2to1[2][2]
    
    return H2to1


def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using RANSAC
    
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches         - matrix specifying matches between these two sets of point locations
        nIter           - number of iterations to run RANSAC
        tol             - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    '''

    ###########################
    # TO DO ...
    

    #H2to1=computeH(np.transpose(locs1[matches[:,0],0:2]),np.transpose(locs2[matches[:,1],0:2]))

    #print(H2to1 / H[0][0])
 
 
    
    no_inlier_list=np.zeros(1)
    
    image_1_list=[]
    image_2_list=[]
   
    for i in range(num_iter):
        #np.random.seed(0)
        matches_rand= np.random.permutation(matches)
    
        subset_indices_1 = matches_rand[0:4,0]
        subset_indices_2 = matches_rand[0:4,1]
    
        subset_1=np.transpose(locs1[subset_indices_1,0:2])
        subset_2= np.transpose(locs2[subset_indices_2,0:2])
        
        
    
        H2to1=computeH(subset_2, subset_1)
    
        set_2 = np.transpose(locs2[matches[:,1] ,0:2])
    
        set_1 =  np.transpose(locs1[matches[:,0] ,0:2])

        set_2_homog= np.vstack((set_2,np.ones((1,set_2.shape[1]))))

        transformed_points_2 = np.matmul(H2to1, set_2_homog)
       
        transformed_points_coord = np.nan_to_num(transformed_points_2 / transformed_points_2[-1,:])[:2,:]
    
        l2_norm = np.sqrt(np.sum(np.nan_to_num((transformed_points_coord - set_1)**2), axis=0))
        
        
        inliers = (l2_norm < tol)
        
        #print(inliers)
        #print(set_1[0][inliers])
        
        no_inliers = np.count_nonzero((inliers))
        
        
        no_inlier_list=np.append(no_inlier_list,no_inliers)
        
        
        image_1_list.append(np.vstack((set_1[0][inliers],set_1[1][inliers])))
        
        image_2_list.append(np.vstack((set_2[0][inliers],set_2[1][inliers])))
        
        #print(no_inliers)

                
        
        #print(inliers)
        
        #print(l2_norm)
        
        #break
        
    #print(max(inlier_list))
    
    
    #print(np.max(no_inlier_list))
    bestH_index=np.argmax(no_inlier_list[1:])
    
    best_image1= image_1_list[bestH_index]
    
    best_image2= image_2_list[bestH_index]
    
    #print(best_image1)
    #print(best_image2)
    
    bestH=computeH(best_image2, best_image1)
    
    #bestH= homography_list[bestH_index]
    
    #print(bestH/bestH[0][0])
    return bestH


if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    ransacH(matches, locs1, locs2, num_iter=5000, tol=2)