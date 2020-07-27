import numpy as np
import multiprocessing
import scipy.ndimage
import skimage
import sklearn.cluster
import scipy.spatial.distance
import os, time
import matplotlib.pyplot as plt
import util
import random
import matplotlib.image as mpimg
#import cv2
#from PIL import Image

def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * filter_responses: numpy.ndarray of shape (H, W, 3F)
    '''

    # ----- TODO -----
   
    if(len(image.shape)!=3):
        image=np.dtsack((image,image,image))
        
    img_lab=skimage.color.rgb2lab(image)
    scales=[1,2,4,8,8*(2**(1/2))]

    for i in range(len(scales)):
      
        if(i==0):
              
            for j in range(3):
                if(j==0):
                    temp1=scipy.ndimage.gaussian_filter(img_lab[:,:,j],sigma=[scales[i],scales[i]],order=0)
                    temp2=scipy.ndimage.gaussian_laplace(img_lab[:,:,j],sigma=[scales[i],scales[i]])
                    temp3=scipy.ndimage.gaussian_filter(img_lab[:,:,j],sigma=[0,scales[i]],order=1)
                    temp4=scipy.ndimage.gaussian_filter(img_lab[:,:,j],sigma=[scales[i],0],order=1)
                else:
                    temp1=np.dstack((temp1,scipy.ndimage.gaussian_filter(img_lab[:,:,j],sigma=[scales[i],scales[i]],order=0)))
                    temp2=np.dstack((temp2,scipy.ndimage.gaussian_laplace(img_lab[:,:,j],sigma=[scales[i],scales[i]])))
                    temp3=np.dstack((temp3,scipy.ndimage.gaussian_filter(img_lab[:,:,j],sigma=[0,scales[i]],order=1)))
                    temp4=np.dstack((temp4,scipy.ndimage.gaussian_filter(img_lab[:,:,j],sigma=[scales[i],0],order=1)))
            
            filter_bank=temp1
            filter_bank=np.dstack((filter_bank,temp2))
            filter_bank=np.dstack((filter_bank,temp3))
            filter_bank=np.dstack((filter_bank,temp4))
        else:
            for j in range(3):
                if(j==0):
                    temp1=scipy.ndimage.gaussian_filter(img_lab[:,:,j],sigma=[scales[i],scales[i]],order=0)
                    temp2=scipy.ndimage.gaussian_laplace(img_lab[:,:,j],sigma=[scales[i],scales[i]])
                    temp3=scipy.ndimage.gaussian_filter(img_lab[:,:,j],sigma=[0,scales[i]],order=1)
                    temp4=scipy.ndimage.gaussian_filter(img_lab[:,:,j],sigma=[scales[i],0],order=1)
                else:
                    temp1=np.dstack((temp1,scipy.ndimage.gaussian_filter(img_lab[:,:,j],sigma=[scales[i],scales[i]],order=0)))
                    temp2=np.dstack((temp2,scipy.ndimage.gaussian_laplace(img_lab[:,:,j],sigma=[scales[i],scales[i]])))
                    temp3=np.dstack((temp3,scipy.ndimage.gaussian_filter(img_lab[:,:,j],sigma=[0,scales[i]],order=1)))
                    temp4=np.dstack((temp4,scipy.ndimage.gaussian_filter(img_lab[:,:,j],sigma=[scales[i],0],order=1)))
            
            filter_bank=np.dstack((filter_bank,temp1))
            filter_bank=np.dstack((filter_bank,temp2))
            filter_bank=np.dstack((filter_bank,temp3))
            filter_bank=np.dstack((filter_bank,temp4))
            
            
                
           
        
    
    #util.display_filter_responses(filter_bank)
    return filter_bank
    
    
   

    pass

def get_visual_words(image, dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * wordmap: numpy.ndarray of shape (H, W)
    '''

    # ----- TODO -----

    filter_bank=extract_filter_responses(image)
    filter_bank_1=np.reshape(filter_bank,(filter_bank.shape[0]*filter_bank.shape[1],filter_bank.shape[2]))
    temp=scipy.spatial.distance.cdist(filter_bank_1,dictionary)
    wordmap_ind=np.argmin(temp,axis=1)    
    wordmap=np.zeros((wordmap_ind.shape[0],1))    
    wordmap=wordmap_ind
    wordmap=np.reshape(wordmap,(image.shape[0],image.shape[1]))
    #out=plt.imshow(wordmap,cmap='jet')
    return wordmap
    
    pass


def compute_dictionary_one_image(args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha, 3F)
    '''


    i, alpha, image_path = args
    # ----- TODO -----
    img1=mpimg.imread("../data/"+image_path)
    img2=img1.astype(float)
    img2=img1/np.max(img2)
    filter_bank=extract_filter_responses(img2)
    filter_bank_1=np.reshape(filter_bank, (filter_bank.shape[0]*filter_bank.shape[1],filter_bank.shape[2]))
    filter_response=np.random.permutation(filter_bank_1)[:alpha,:]
    filename="../temporary/" + str(i) + ".txt"
    np.savetxt(filename,filter_response)
   
    return filter_response
    

    pass

def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * dictionary: numpy.ndarray of shape (K, 3F)
    '''

    train_data = np.load("../data/train_data.npz")    
    training_files=train_data['files']
    training_labels=train_data['labels']
    alpha=150
    K=100
    compute_dictionary_one_image([0,alpha,training_files[0]])
    for i in range(len(training_files)):
        print(i)
        if(i==0):
            training_image_responses=compute_dictionary_one_image([i,alpha,training_files[i]])
        else:
            training_image_responses=np.vstack((training_image_responses,compute_dictionary_one_image([i,alpha,training_files[i]])))

    kmeans=sklearn.cluster.KMeans(n_clusters=K,n_jobs=-1).fit(training_image_responses)
    dictionary=kmeans.cluster_centers_
    np.save("dictionary.npy",dictionary)

        
        
    
    
    
    # ----- TODO -----

    pass





