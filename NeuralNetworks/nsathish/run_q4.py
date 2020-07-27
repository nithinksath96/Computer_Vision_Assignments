
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

''' Visualize training data '''
# import scipy.io
# train_data = scipy.io.loadmat('../data/nist36_train.mat')

# train_x, train_y = train_data['train_data'], train_data['train_labels']

# for i in range(train_x.shape[0]):
#     single_x = train_x[i+int(3*train_x.shape[0] / 36)]
#     single_x = single_x.reshape(32, 32)
#     plt.imshow(single_x, cmap='gray')
#     plt.show()

bbox_padding_x = 20
bbox_padding_y = 18

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)
    

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    num_letters = len(bboxes)
    height = bw.shape[0]
    width = bw.shape[1]

    letters = np.empty((0, 1024))
    row_index = np.zeros((num_letters, 1))

    def centroid_y(bbox):
        y1, x1, y2, x2 = bbox
        return (y2 + y1) / 2
    sorted(bboxes, key=centroid_y)

    yc_last = 0
    curr_row_id = 0
    bboxes_sorted = []

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        y1, x1, y2, x2 = bbox
        yc = (y2 + y1) / 2
        xc = (x2 + x1) / 2

        if (yc - yc_last) >= (y2 - y1):
            curr_row_id += 1
        yc_last = yc
        row_index[i] = curr_row_id
        bboxes_sorted.append((y1, x1, y2, x2, curr_row_id))

    # sort based on row_index first, and then centroid_x position (from left to right)
    bboxes_sorted = sorted(bboxes_sorted, key=lambda x: (x[-1], (x[1] + x[3] / 2)))

    for i in range(len(bboxes_sorted)):
        bbox = bboxes_sorted[i]
        y1, x1, y2, x2, row_id = bbox
        yc = (y2 + y1) / 2
        xc = (x2 + x1) / 2

        # print("Centroid y: ", yc )
        # print("Centroid x: ", xc )
        # print("Row index: ", row_id )
        y_centroid = (y2 - y1) / 2
        x_centroid = (x2 - x1) / 2

        y1 = max(0, y1 - bbox_padding_y)
        x1 = max(0, x1 - bbox_padding_x)
        y2 = min(height, y2 + bbox_padding_y)
        x2 = min(width, x2 + bbox_padding_x)
        
        bw=bw*1.0
        letter = bw[y1:y2+1, x1:x2+1]
        letter = skimage.morphology.binary_erosion(letter)
        letter = skimage.filters.gaussian(letter, sigma=1.0)
        letter_square = skimage.transform.resize(letter, (32, 32)).transpose()
        # letter_square = letter_square.T
        letter_flattened = letter_square.reshape(-1)

        letters = np.append(letters, letter_flattened.reshape((1, 1024)), axis=0)

        #plt.imshow(letter_square, cmap='gray')
        #plt.show()
    print("\nCurrent image: " + img)
    # print("Letters shape: ", letters.shape)
    print("Num letters in the image: ", num_letters)

    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    # letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))

    h1 = forward(letters, params, 'layer1')
    probs = forward(h1, params, 'output', softmax)

    # ground_truth = np.argmax(test_y, axis=1)
    predicted = np.argmax(probs, axis=1)
    # print(predicted)
    letter_list = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    predicted_letters = letter_list[predicted]

    num_extracted_letters = predicted_letters.shape[0]
    
    curr_row = ""
    curr_row_id = 0
    for i in range(num_extracted_letters):
        if row_index[i] == curr_row_id:
            curr_row += predicted_letters[i]
        else:
            print(curr_row)
            curr_row = "" + predicted_letters[i]
            curr_row_id = row_index[i]
    print(curr_row)
