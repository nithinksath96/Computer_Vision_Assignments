#!/usr/bin/env python
# coding: utf-8

# In[58]:


import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils import data
from torchvision import transforms
#from torchvision.datasets import MNIST,EMNIST

import matplotlib.pyplot as plt
import time
import scipy.io


# In[59]:


cuda = torch.cuda.is_available()
cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[60]:


torchvision.datasets.EMNIST.url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'
train_dataset = torchvision.datasets.EMNIST(root='../../data',
                                            split='balanced',
                                            train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.EMNIST(root='../../data',
                                           split='balanced',
                                           train=False,
                                          transform=transforms.ToTensor())


# In[61]:


train_x = train_dataset.data
train_y = train_dataset.targets

test_x = test_dataset.data
test_y = test_dataset.targets
#print(train_x.shape)
#print(train_y.shape)
#print(test_x.shape)
#print(test_y.shape)


# In[62]:


class MyDataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self,index):
        X = self.X[index].float().reshape(-1) 
        Y = self.Y[index].long()
        return X,Y


# In[63]:


class FC_model(nn.Module):
    def __init__(self):
        super(FC_model,self).__init__()
        layers = []
        layers.append(nn.Linear(1024,64))
        layers.append(nn.Sigmoid())
        layers.append(nn.Linear(64,36))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
FC_model()


# In[64]:


class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model,self).__init__()
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 47)
    
    def forward(self, x):
        
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        x = x.view(-1, 320)
        
        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        
        return x
    
    


# In[65]:


model = CNN_model()
criterion = nn.CrossEntropyLoss()
learning_rate = 1e-2
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
def init_xavier(m):
    if type(m) == nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0,std)

model.apply(init_xavier)


# In[66]:


# g0 = np.random.multivariate_normal([3.6,40],[[0.05,0],[0,10]],10)
# g1 = np.random.multivariate_normal([3.9,10],[[0.01,0],[0,5]],10)
# g2 = np.random.multivariate_normal([3.4,30],[[0.25,0],[0,5]],10)
# g3 = np.random.multivariate_normal([2.0,10],[[0.5,0],[0,10]],10)
# train_x = np.vstack([g0,g1,g2,g3])
# # we will do XW + B
# # that implies that the data is N x D

# # create labels
# y_idx = np.array([0 for _ in range(10)] + [1 for _ in range(10)] + [2 for _ in range(10)] + [3 for _ in range(10)])
# # turn to one_hot
# y = np.zeros((y_idx.shape[0],y_idx.max()+1))
# y[np.arange(y_idx.shape[0]),y_idx] = 1
# train_y=y


# In[67]:


#train_data = torch.tensor(train_x,dtype=torch.float)
#train_targets=torch.tensor(train_y, dtype=torch.long)
#train_dataset = MyDataset(train_x, train_y)

# valid_data = torch.tensor(valid_x,dtype=torch.float)
# valid_targets=torch.tensor(valid_y, dtype=torch.long)
#valid_dataset = MyDataset(test_x, test_y)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False)



# train_loader_args = dict(shuffle=True, batch_size=64, pin_memory=True) 
# train_loader = data.DataLoader(train_dataset, **train_loader_args)

# valid_loader_args = dict(shuffle=False, batch_size=64, pin_memory=True) 
# valid_loader = data.DataLoader(valid_dataset, **valid_loader_args)


# In[68]:


def train_epoch(model, train_loader, criterion, optimizer):
    #model.train()
    #model.to(device)
    running_loss = 0.0
    total_acc =0.0

    
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        

        
        optimizer.zero_grad()

        outputs = model(data)

        _, predicted = torch.max(outputs.data, 1)

        total_acc+= ((target==predicted).sum().item())
        loss = criterion(outputs,target)
        running_loss += loss.item()
        
        
        loss.backward()
        optimizer.step()
        
    end_time = time.time()
    
    avg_acc = total_acc / train_x.shape[0]
    return running_loss,avg_acc

#train_epoch(model, train_loader, criterion, optimizer)


# In[69]:


def test_model(model, test_loader, criterion):
    with torch.no_grad():
        model.eval()
        #model.to(device)

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0

        for batch_idx, (data, target) in enumerate(test_loader):   


            outputs = model(data)

            
            total_predictions += target.size(0)
            
            
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == target).sum().item()

            loss = criterion(outputs, target).detach()
            running_loss += loss.item()


        
        acc = (correct_predictions/test_x.shape[0])

        return running_loss, acc
#test_model(model, valid_loader, criterion)


# In[70]:


n_epochs=5
TrainLoss=[]
TrainAcc=[]
ValidLoss=[]
ValidAcc=[]
TestLoss=[]
TestAcc=[]
for i in range(n_epochs):
    train_loss,avg_acc = train_epoch(model, train_loader, criterion, optimizer)
    valid_loss,valid_acc = test_model(model, test_loader, criterion)
    TrainAcc.append(avg_acc)
    TrainLoss.append(train_loss)
    ValidAcc.append(valid_acc)
    ValidLoss.append(valid_loss)
    
    if(i%1==0):
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(i,train_loss, avg_acc))
        print(" Validation loss: {:.2f} \t Validation acc : {:.2f}".format(valid_loss, valid_acc))
    #print(train_loss)
    #break


# In[71]:


plt.figure(0)
plt.title('Training Loss and validation loss')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(TrainLoss , 'r')
plt.plot(ValidLoss, 'b')
plt.legend(['training loss','validation loss'])
plt.show()


# In[30]:


plt.figure(0)
plt.title('Training ACC and validation ACC')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(TrainAcc , 'r')
plt.plot(ValidAcc, 'b')
plt.legend(['training ACC','validation ACC'])

plt.show()


# In[76]:



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
from testq4 import *
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

    letters = np.empty((num_letters,1,28,28))
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
        letter_square = skimage.transform.resize(letter, (28,28)).transpose()
        # letter_square = letter_square.T
        #letter_flattened = letter_square.reshape(-1)

        #letters = np.append(letters, letter_flattened.reshape((1, 1024)), axis=0)
        letters[i,0,:,:] = letter_square
    import pickle
    import string
    letters_list = np.array([str(_) for _ in range(10)] +
                   [_ for _ in string.ascii_uppercase[:26]]+ ["a","b","d","e","f","g","h","n","q","r","t"]  )
    #params = pickle.load(open('q3_weights.pickle','rb'))
    ##########################
    ##### your code here #####
    ##########################
    letters = torch.tensor(letters).float()
    out = model(letters)
    _, predicted = torch.max(out.data, 1)
    
    predicted_letters = letters_list[predicted]

    num_extracted_letters = predicted_letters.shape[0]
    
    curr_row = ""
    curr_row_id = 0
    for i in range(num_extracted_letters):
        if row_index[i] == curr_row_id:
            curr_row += predicted_letters[i]
        else:
            #print(curr_row)
            curr_row = "" + predicted_letters[i]
            curr_row_id = row_index[i]
    #print(curr_row)


# In[ ]:




