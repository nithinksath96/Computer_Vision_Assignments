import numpy as np
import multiprocessing
import threading
import queue
import os,time
import torch
import skimage.transform
import torchvision.transforms
import torch.nn as nn
import util
import matplotlib.image as mpimg
import scipy.spatial.distance
#import network_layers

def build_recognition_system(vgg16, num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N, K)
    * labels: numpy.ndarray of shape (N)
    '''

    train_data = np.load("../data/train_data.npz")
    
    train_path= train_data['files']
    train_labels= train_data['labels']
#    for i in range(int(len(train_path))):
#       
#        if(i==0):
#            features= get_image_feature([i,train_path[i],vgg16]).detach().numpy()
#        
#        else:
#            features=np.vstack((features,get_image_feature([i,train_path[i],vgg16]).detach().numpy()))
#            
# 
    features=np.load("feature.npy")
 
        
    np.savez('trained_system_deep.npz', name1=features , name2= train_labels)

    # ----- TODO -----
    
    pass
    

def evaluate_recognition_system(vgg16, num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8, 8)
    * accuracy: accuracy of the evaluated system
    '''
    train_data = np.load("../data/train_data.npz")
    trained_system_deep = np.load("trained_system_deep.npz")
    train_features= trained_system_deep['name1']
    train_labels= train_data['labels']
    
    test_data = np.load("../data/test_data.npz")
    test_path= test_data['files']
    test_labels=test_data['labels']
  
    predicted_label=np.zeros(test_labels.shape[0])
    correct=0
    for i in range(int(len(test_path))):
      
     
        feature=get_image_feature([i,test_path[i],vgg16])
        dist=distance_to_set(feature.detach().numpy(),train_features)
        arg1=np.argmin(dist)
      
        predicted_label[i]= train_labels[arg1]

    correct=np.count_nonzero(predicted_label==test_labels)
    accuracy=correct/len(test_labels)
    
    
    
    classes=len(set(test_labels))
    conf_matrix = np.zeros((classes,classes))
    
    for i in range(test_labels.shape[0]):

        conf_matrix[int(test_labels[i])][int(predicted_label[i])]+=1
    
    return accuracy, conf_matrix
    

    
        #break
        

    # ----- TODO -----
    
    pass


def preprocess_image(image):
    '''
    Preprocesses the image to load into the prebuilt network.

    [input]
    * image: numpy.ndarray of shape (H, W, 3)

    [output]
    * image_processed: torch.Tensor of shape (3, H, W)
    '''

    # ----- TODO -----
    mean=[0.485,0.456,0.406]
    std=[0.229,0.224, 0.225]
 
    image=skimage.transform.resize(image,(224,224,3))
    for i in range(image.shape[2]):
        image[:,:,i]=(image[:,:,i] - mean[i])/std[i]
        
    image_pre=np.array([[image[:,:,0],image[:,:,1],image[:,:,2]]])
    
    image_pre=torch.Tensor(image_pre)
    

    return image_pre
    #pass


def get_image_feature(args):
    '''
    Extracts deep features from the prebuilt VGG-16 network.
    This is a function run by a subprocess.
    [input]
    * i: index of training image
    * image_path: path of image file
    * vgg16: prebuilt VGG-16 network.
    
    [output]
    * feat: evaluated deep feature
    '''

    i, image_path, vgg16 = args
    image=mpimg.imread("../data/" + image_path)

    if(len(image.shape)!=3):
        image=np.dstack((image,image,image))
    image_pre=preprocess_image(image)

    
    
    
    features=vgg16(image_pre)
   
    return features    

    # ----- TODO -----
    
    pass




def distance_to_set(feature, train_features):
    '''
    Compute distance between a deep feature with all training image deep features.

    [input]
    * feature: numpy.ndarray of shape (K)
    * train_features: numpy.ndarray of shape (N, K)

    [output]
    * dist: numpy.ndarray of shape (N)
    '''
    

    dist=scipy.spatial.distance.cdist(train_features,feature,metric='euclidean')

    return dist
    pass



#vgg16=torchvision.models.vgg16(pretrained=True)
#vgg16.classifier=nn.Sequential(*list(vgg16.classifier.children())[:-3])
#build_recognition_system(vgg16)
#evaluate_recognition_system(vgg16)
