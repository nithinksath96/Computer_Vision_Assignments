# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:08:47 2019

@author: nithi
"""

    correct=np.count_nonzero(predicted_labels==test_labels)
    accuracy=correct/len(test_labels)
    print(accuracy)
    #print(len(set(test_labels)))
    #print(predicted_labels)
    classes=len(set(test_labels))
    conf_matrix = np.zeros((classes,classes))
    print(conf_matrix.shape)
    for i in range(test_labels.shape[0]):

        conf_matrix[int(test_labels[i])][int(predicted_labels[i])]+=1

    print("one",sklearn.metrics.confusion_matrix(test_labels,predicted_labels))
    print("two",conf_matrix)
