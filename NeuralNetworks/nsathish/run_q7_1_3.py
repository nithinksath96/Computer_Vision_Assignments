#!/usr/bin/env python
# coding: utf-8

# In[55]:


import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils import data
from torchvision import transforms
from torchvision.datasets import MNIST

import matplotlib.pyplot as plt
import time
import scipy.io


# In[56]:


cuda = torch.cuda.is_available()
cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[57]:


train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

print(train_x.shape)
print(train_x[0])


# In[58]:


class MyDataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self,index):
        X = self.X[index].float().reshape(1,32,32)
        Y = self.Y[index].long()
        return X,Y


# In[59]:


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


# In[60]:


class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model,self).__init__()
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 36)
    
    def forward(self, x):
        #print("FORWRD1",x.shape)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #print("FORWRD2",x.shape)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        #print("FORWRD3",x.shape)
        x = x.view(-1, 500)
        #print("FORWRD4",x.shape)
        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        
        return x
    
    


# In[61]:


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


# In[62]:


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


# In[63]:


train_data = torch.tensor(train_x,dtype=torch.float)
train_targets=torch.tensor(train_y, dtype=torch.long)
train_dataset = MyDataset(train_data, train_targets)

valid_data = torch.tensor(valid_x,dtype=torch.float)
valid_targets=torch.tensor(valid_y, dtype=torch.long)
valid_dataset = MyDataset(valid_data, valid_targets)



train_loader_args = dict(shuffle=True, batch_size=16, pin_memory=True) 
train_loader = data.DataLoader(train_dataset, **train_loader_args)

valid_loader_args = dict(shuffle=False, batch_size=16, pin_memory=True) 
valid_loader = data.DataLoader(valid_dataset, **valid_loader_args)


# In[64]:


def train_epoch(model, train_loader, criterion, optimizer):
    #model.train()
    #model.to(device)
    running_loss = 0.0
    total_acc =0.0
    print(running_loss)
    
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        #data=data.to(device)
        
        #print("--------------------------------",data.shape)
        
        optimizer.zero_grad() 
        label = np.where(target == 1)[1]
        label = torch.tensor(label)
        outputs = model(data)
        #print(target)
        #print(label)
        #print("++++++++++++++++++++++++++++++++++",outputs.shape)
        _, predicted = torch.max(outputs.data, 1)
        #print(predicted.shape)
        #label = label.to(device)
        total_acc+= ((label==predicted).sum().item())
        loss = criterion(outputs, label)
        running_loss += loss.item()
        
        
        loss.backward()
        optimizer.step()
        
    end_time = time.time()
    
    avg_acc = total_acc / train_x.shape[0]
    return running_loss,avg_acc

#train_epoch(model, train_loader,criterion,optimizer)


# In[65]:


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
            
            
            label = np.where(target == 1)[1]
            label = torch.tensor(label)
            
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == label).sum().item()

            loss = criterion(outputs, label).detach()
            running_loss += loss.item()


        
        acc = (correct_predictions/valid_x.shape[0])

        return running_loss, acc
#test_model(model, valid_loader, criterion)


# In[68]:


n_epochs=10
TrainLoss=[]
TrainAcc=[]
ValidLoss=[]
ValidAcc=[]
TestLoss=[]
TestAcc=[]
for i in range(n_epochs):
    train_loss,avg_acc = train_epoch(model, train_loader, criterion, optimizer)
    valid_loss,valid_acc = test_model(model, valid_loader, criterion)
    TrainAcc.append(avg_acc)
    TrainLoss.append(train_loss)
    ValidAcc.append(valid_acc)
    ValidLoss.append(valid_loss)
    
    if(i%1==0):
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(i,train_loss, avg_acc))
        print(" Validation loss: {:.2f} \t Validation acc : {:.2f}".format(valid_loss, valid_acc))
    #print(train_loss)
    #break


# In[69]:


plt.figure(0)
plt.title('Training Loss and validation loss')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(TrainLoss , 'r')
plt.plot(ValidLoss, 'b')
plt.legend(['training loss','validation loss'])
plt.show()


# In[70]:


plt.figure(0)
plt.title('Training ACC and validation ACC')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(TrainAcc , 'r')
plt.plot(ValidAcc, 'b')
plt.legend(['training ACC','validation ACC'])

plt.show()


# In[ ]:




