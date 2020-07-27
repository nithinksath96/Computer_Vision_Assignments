from alignChannels import alignChannels
import numpy as np
import cv2 as cv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.ndimage
# Problem 1: Image Alignment

# 1. Load images (all 3 channels)
red = np.load("../data/red.npy")
green = np.load("../data/green.npy")
blue = np.load("../data/blue.npy")
#print(red.shape)
#print(green.shape)
#print(blue.shape)
print("Red",red)
#plt.imshow(red)
#plt.imshow(green)
#t.imshow(blue)
#plt.show()
#plt.show()
#plt.show()
#blue_cropped= blue[300:600, 300:600]
#red_cropped=  red[300:600, 300:600]

#print(red)
#print(green)
#print(blue)
#cv.imshow("Green",green)
#cv.imshow("Red",red)
#cv.imshow("Blue",blue)
#temp=np.roll(blue,(-30),axis=1)
#blue_f=np.roll(blue,(-30)*blue.shape[1])

# temp=np.roll(blue,(-30),axis=1)
# blue_f=np.roll(blue,(-30)*blue.shape[1])


#red_f= scipy.ndimage.shift(red,(+10,0))
#green_f=scipy.ndimage.shift(green,(-30,0))



#rgb_output=np.dstack((red_f,green_f,blue))
#cv.imshow("Output",rgb_output)
#cv.waitKey(0)
#np.savetxt("C:\\Users\\nithi\\Desktop\\cmu\\Computer_Vision\\hw0\\data\\red.txt",red)
#np.savetxt("C:\\Users\\nithi\\Desktop\\cmu\\Computer_Vision\\hw0\\data\\blue.txt",blue)
#np.savetxt("C:\\Users\\nithi\\Desktop\\cmu\\Computer_Vision\\hw0\\data\\green.txt",green)
# 2. Find best alignment
rgbResult = alignChannels(red, green, blue)
#print(rgbResult.shape)
#out=np.reshape(rgbResult,(810,943,3))
#print(out.shape)
cv.imwrite("../results/rgb_output.jpg",rgbResult)
# 3. save result to rgb_output.jpg (IN THE "results" FOLDER)
