import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation


import matplotlib.image as mpimg
from skimage.filters import threshold_otsu , threshold_mean, threshold_local
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square , opening,convex_hull_image
from skimage.color import label2rgb,rgb2gray
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    ##########################
    ##### your code here #####
    ##########################
    

#   
    image_gray = rgb2gray(image)
    blurred = skimage.filters.gaussian(image_gray, sigma=1.0)

    thresh = threshold_otsu(blurred)
    bn1 = image_gray < thresh
    bw= image_gray > thresh

    bn = opening(bn1)

    label_image = label(bn)

    image_label_overlay = label2rgb(label_image, image=image)

    regions = regionprops(label_image)
#    
    total_area = 0
    for region in regions:
        total_area += region.area
    mean_area = total_area / len(regions)
    
    bboxes = []
    for region in regions:
        # take regions with large enough areas
        if region.area >= mean_area / 2:
            # draw rectangle around segmented letters
            min_row, min_col, max_row, max_col = region.bbox

            bboxes.append(np.array([min_row, min_col, max_row, max_col]))
    
    

    return bboxes, bw

