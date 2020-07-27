# -*- coding: utf-8 -*-
from BRIEF import briefLite, briefMatch
import numpy as np
import cv2
from matplotlib import pyplot as plt
from keypointDetect import DoGdetector

im1 = cv2.imread('../data/model_chickenbroth.jpg')
#im2 = cv2.imread('../data/model_chickenbroth.jpg')

#im2 = cv2.imread('../data/pf_stand.jpg')
locs1, desc1 = briefLite(im1)
center=(im1.shape[0]/2, im1.shape[1]/2)
#print(center)
correct_matches=[]
for i in range(36):
    rotated_matrix=cv2.getRotationMatrix2D(center,i*10,1)
    out= cv2.warpAffine(im1,rotated_matrix,(im1.shape[0],im1.shape[1]))
    locs2, desc2 = briefLite(out)
    matches = briefMatch(desc1, desc2)
    #print(len(matches))
    correct_matches.append(len(matches))
    
    #print(len(matches))
    #cv2.imshow("Rotated",out)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #break
    
#locs2, desc2 = briefLite(im2)
#matches = briefMatch(desc1, desc2)
#plt.show()
x=range(0,360,10)
#plt.bar(x,correct_matches)
#plt.xlabel("Rotation angle")
#plt.ylabel("Correct matches")
#plt.title("No of correct matches vs rotation angle ")
#plt.show()