#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import MNIST

import matplotlib.pyplot as plt
import time
import scipy.io


# In[11]:


cuda = torch.cuda.is_available()
cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[12]:


custom_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
train_dataset = torchvision.datasets.ImageFolder(root='../../data/oxford-flowers17/train/',transform=custom_transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, 
                                               shuffle=True)
test_dataset = torchvision.datasets.ImageFolder(root='../../data/oxford-flowers17/test/', 
                                               transform=custom_transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, 
                                             shuffle=False)

print(len(train_dataset))

#root='train_data/medium/'


# In[42]:


class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model,self).__init__()
        
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 50, kernel_size=5)
        self.conv3 = nn.Conv2d(50, 200, kernel_size=5)
        self.fc1 = nn.Linear(24*24*200, 500)
        self.fc2 = nn.Linear(500, 17)
    
    def forward(self, x):
        
        #print(x.shape)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        
        x = F.relu(F.max_pool2d(self.conv3(x), 2))

        x = x.view(-1, 24*24*200)
        
        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        
        return x
    
    


# In[21]:


model = torchvision.models.squeezenet1_1(pretrained = True)
#print(*list(model.classifier.children())[:])
#print(model)


# In[22]:


#model.final_layer = torch.nn.Linear(512, 17)
# torch.nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
# model = torchvision.models.squeezenet1_1(pretrained=True)
# model.classifier = torch.nn.Sequential(
#     torch.nn.Dropout(p=0.2),
#     final_layer,
#     torch.nn.ReLU(inplace=True),
#     torch.nn.AvgPool2d(13)
# )
# model.forward = lambda x: model.classifier(model.features(x)).view(x.size(0), 17)


# In[24]:


final_layer = torch.nn.Conv2d(512, 17, kernel_size=1)
torch.nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
model = torchvision.models.squeezenet1_1(pretrained=True)
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2),
    final_layer,
    torch.nn.ReLU(inplace=True),
    torch.nn.AvgPool2d(13)
)
model.forward = lambda x: model.classifier(model.features(x)).view(x.size(0), 17)


# In[25]:


#model = CNN_model()
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


# In[26]:


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
    
    avg_acc = total_acc / len(train_dataset)
    return running_loss,avg_acc

#train_epoch(model, train_loader, criterion, optimizer)


# In[27]:


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


# In[28]:


n_epochs=5
TrainLoss=[]
TrainAcc=[]
ValidLoss=[]
ValidAcc=[]
TestLoss=[]
TestAcc=[]
for i in range(n_epochs):
    train_loss,avg_acc = train_epoch(model, train_dataloader, criterion, optimizer)
    #valid_loss,valid_acc = test_model(model, test_dataloader, criterion)
    TrainAcc.append(avg_acc)
    TrainLoss.append(train_loss)
    #ValidAcc.append(valid_acc)
    #ValidLoss.append(valid_loss)
    
    if(i%1==0):
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(i,train_loss, avg_acc))
        #print(" Validation loss: {:.2f} \t Validation acc : {:.2f}".format(valid_loss, valid_acc))
    #print(train_loss)
    #break


# In[29]:


plt.figure(0)
plt.title('Training Loss')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(TrainLoss , 'r')

plt.legend(['training loss'])
plt.show()


# In[30]:


plt.figure(0)
plt.title('Training ACC')
plt.xlabel('Epoch Number')
plt.ylabel('Acc')
plt.plot(TrainAcc , 'r')

plt.legend(['training ACC'])

plt.show()


# In[ ]:




