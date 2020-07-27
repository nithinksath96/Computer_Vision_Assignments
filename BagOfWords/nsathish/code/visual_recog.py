import numpy as np
import skimage
import multiprocessing
import threading
import queue
import os,time
import math
import visual_words
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def build_recognition_system(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N, M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K, 3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")
    # ----- TODO -----
    
    training_files=train_data['files']
    training_labels=train_data['labels']
    for i in range(len(training_files)):
#    
#        print(i)
        if(i==0):
            features=get_image_feature(training_files[i],dictionary,3,dictionary.shape[0])
            #print(features)
#        else:
#            histograms = get_image_feature(training_files[i],dictionary,3,dictionary.shape[0])
#            features=np.vstack((features, histograms))
#            
#            
#    np.save("feature_bow.npy",features)
    features=np.load("feature_bow.npy")
    
    np.savez('trained_system.npz', name1=dictionary, name2=features , name3= training_labels, name4= 3)

    pass


def evaluate_recognition_system(num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8, 8)
    * accuracy: accuracy of the evaluated system
    '''


    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system.npz")

    features= trained_system['name2']
    layer= trained_system['name4']
    train_labels= trained_system['name3']

    dictionary= trained_system['name1']
    K=dictionary.shape[0]
    test=test_data['files']
    test_labels= test_data['labels']
    predicted_label=np.zeros(test_labels.shape[0])
    for i in range(int(len(test))):

        img1=mpimg.imread("../data/"+test[i])
        img2=img1.astype(float)
        img2=img1/np.max(img2)
        if(len(img2.shape)!=3):
            img2=np.dtsack((img2,img2,img2))


        wordmap=visual_words.get_visual_words(img2,dictionary)
        
        histograms_test=get_feature_from_wordmap_SPM(wordmap,layer,K)
        #print(histograms_test)
        #break
        histograms_test=np.reshape(histograms_test,(1,histograms_test.shape[0]))
        nearest_match=distance_to_set(features,histograms_test)
        print("Nearest match",nearest_match)
        predicted_label[i]=train_labels[nearest_match]
      

    
    correct=np.count_nonzero(predicted_label==test_labels)
    accuracy=correct/len(test_labels)
    
    classes=len(set(test_labels))
    conf_matrix = np.zeros((classes,classes))
   
    for i in range(test_labels.shape[0]):

        conf_matrix[int(test_labels[i])][int(predicted_label[i])]+=1
    
    
    return accuracy, conf_matrix
        
        
    # ----- TODO -----
    
    
    

    pass


def get_image_feature(file_path, dictionary, layer_num, K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    # ----- TODO -----
    img1=mpimg.imread("../data/"+file_path)
    img2=img1.astype(float)
    img2=img1/np.max(img2)
    if(len(img2.shape)!=3):
        img2=np.dtsack((img2,img2,img2))
   

    wordmap=visual_words.get_visual_words(img2,dictionary)
    
    histograms=get_feature_from_wordmap_SPM(wordmap,layer_num,K)
    histograms=np.reshape(histograms,(1,histograms.shape[0]))
    
    return histograms
    
    
    
    pass


def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N, K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    # ----- TODO -----
    
    minimum_val=np.minimum(word_hist,histograms)
    print("histograms",histograms.shape)
    sum_minimum= np.sum(minimum_val,axis=1)
    print("Sum",sum_minimum.shape)
    nearest_match=np.argmax(sum_minimum)
    return nearest_match

    pass


def get_feature_from_wordmap(wordmap, dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    # ----- TODO -----
    #print(dict_size)
    vals,edges=np.histogram(wordmap,bins=dict_size, density=False)

    y=list(vals)
    x=range(dict_size)
    plt.plot(x,y)
    pass


def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    
    map_list=[[wordmap]]
    for i in range(layer_num-1):
        map_1= map_list[i]
        temp=[]
   
        for j in range(len(map_1)):
        
        #axis=1 split vertically axis=0 split horizontally
            left,right=np.array_split(map_1[j],2,axis=1)
            top_left,bottom_left= np.array_split(left,2,axis=0)
            top_right,bottom_right=np.array_split(right,2,axis=0)
            temp.append(top_left)
            temp.append(top_right)
            temp.append(bottom_left)
            temp.append(bottom_right)
        map_list.append(temp)
    

    vals_list=[]
    l=len(map_list)
    
    weight_constant= 2.0**(l-layer_num-1)
    for i in range(len(map_list[-1])):
        w=weight_constant*(np.ones((map_list[-1][i].shape[0],map_list[-1][i].shape[1])))
        vals,edges=np.histogram(map_list[-1][i],bins=dict_size,weights=w, density=False)
        vals_list.append(vals)
    
    
    
    vals_list_2=[vals_list]    
    for i in range(len(map_list)-1):
        w_map= vals_list_2[i]
        temp=[]
        if(i>=(len(map_list)-2)):
            wc=2.0**(-(layer_num-2))
        else:
            wc=2.0**(-(i+1))
        for j in range(0,len(w_map)-3,4):
            t = wc*(w_map[j] + w_map[j+1] + w_map[j+2] + w_map[j+3])
            temp.append(t)

        vals_list_2.append(temp)

    hist=np.zeros(1)
    for i in range(len(vals_list_2)):
        words=np.array(vals_list_2[i]).flatten()
        words_normalized=words/np.sum(words)

        hist=np.append(hist,words_normalized)
        
    hist=hist[1:]

    hist_normalized=hist/len(hist)
    return hist_normalized 
    
        
        

    # ----- TODO -----
    
    pass



print(evaluate_recognition_system())
#print(build_recognition_system())



