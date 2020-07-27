import scipy
import numpy as np
import scipy.ndimage
import os
import util
import skimage
import matplotlib.image as mpimg

def extract_deep_feature(x, vgg16_weights):
    '''
    Extracts deep features from the given VGG-16 weights.

    [input]
    * x: numpy.ndarray of shape (H, W, 3)
    * vgg16_weights: list of shape (L, 3)

    [output]
    * feat: numpy.ndarray of shape (K)
    '''

    output=x
    for i in range(31):
        if(weights[i][0]=='conv2d'):
            
            output=multichannel_conv2d(output,vgg16_weights[i][1],vgg16_weights[i][2])

        
        elif(weights[i][0]=='relu'):
           
            output=relu(output)
        
        elif(weights[i][0]=='maxpool2d'):
            
           
            
            output=max_pool2d(output, vgg16_weights[i][1])
           
        else:
            
            
            output=output.flatten()
            output=linear(output, vgg16_weights[i][1], vgg16_weights[i][2])
    

    return output
    
    
    pass


def multichannel_conv2d(x, weight, bias):
    '''
    Performs multi-channel 2D convolution.

    [input]
    * x: numpy.ndarray of shape (H, W, input_dim)
    * weight: numpy.ndarray of shape (output_dim, input_dim, kernel_size, kernel_size)
    * bias: numpy.ndarray of shape (output_dim)

    [output]
    * feat: numpy.ndarray of shape (H, W, output_dim)
    '''
   
    
    for i in range(weight.shape[0]):
        temp=np.zeros((x.shape[0],x.shape[1]))
        for j in range(weight.shape[1]):
            
            temp+=scipy.ndimage.convolve(x[:,:,j],weight[i][j][::-1,::-1], mode='constant',cval=0)
          
        
        temp= temp+ bias[i]
       
        if(i==0):
            output=temp 
        else:
            output=np.dstack((output,temp))
    
    return output
        
        

    #pass

def relu(x):
    '''
    Rectified linear unit.

    [input]
    * x: numpy.ndarray

    [output]
    * y: numpy.ndarray
    '''
    
    return np.maximum(x,0)
    #pass

def max_pool2d(x, size):
    '''
    2D max pooling operation.

    [input]
    * x: numpy.ndarray of shape (H, W, input_dim)
    * size: pooling receptive field

    [output]
    * y: numpy.ndarray of shape (H/size, W/size, input_dim)
    '''
    maxpool_output=np.zeros((int(x.shape[0]/size),int(x.shape[1]/size),x.shape[2]))
    for i in range(x.shape[2]):
        channel= x[:,:,i]
        for j in range(int(x.shape[0]/size)):
            for k in range(int(x.shape[1]/size)):
            
                maxpool_output[j][k][i]=np.max(channel[j*size:j*size+size,k*size:k*size+size])
    
    return maxpool_output  
            
    #pass

def linear(x,W,b):
    '''
    Fully-connected layer.

    [input]
    * x: numpy.ndarray of shape (input_dim)
    * weight: numpy.ndarray of shape (output_dim,input_dim)
    * bias: numpy.ndarray of shape (output_dim)

    [output]
    * y: numpy.ndarray of shape (output_dim)
    '''
    linear_output=np.matmul(W,x) + b
    return linear_output
    #pass

img1=mpimg.imread("../data/aquarium/sun_aairflxfskjrkepm.jpg")

if(len(img1.shape)!=3):
    img1=np.dtsack((img1,img1,img1))

image=skimage.transform.resize(img1,(224,224,3  ))

weights=util.get_VGG16_weights()

mean=[0.485, 0.456 , 0.406]
std=[0.229, 0.224, 0.225]

for i in range(image.shape[2]):
    image[:,:,i]=(image[:,:,i] - mean[i])/std[i]


extract_deep_feature(image, weights)


