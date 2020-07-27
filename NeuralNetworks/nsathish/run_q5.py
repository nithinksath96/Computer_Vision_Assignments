import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

print(train_x.shape)

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
##########################
##### your code here #####
##########################
initialize_weights(1024,32,params,'layer1')
initialize_weights(32,32,params,'layer2')
initialize_weights(32,32,params,'layer3')
initialize_weights(32,1024,params,'output')

params["Momentum"+"W"+"layer1"] = np.zeros((1024,32))
params["Momentum"+"W"+"layer2"] = np.zeros((32,32))
params["Momentum"+"W"+"layer3"] = np.zeros((32,32))
params["Momentum"+"W"+"output"] = np.zeros((32,1024))

params["Momentum"+"b"+"layer1"] = np.zeros(32)
params["Momentum"+"b"+"layer2"] = np.zeros(32)
params["Momentum"+"b"+"layer3"] = np.zeros(32)
params["Momentum"+"b"+"output"] = np.zeros(1024)

losses_list=[]

# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        ##########################
        ##### your code here #####
        ##########################
        h1 = forward(xb,params,'layer1',relu)
        h2 = forward(h1,params,'layer2',relu)
        h3 = forward(h2,params,'layer3',relu)
        probs = forward(h3,params,'output',sigmoid)
        #print(probs)
        loss = np.sum((probs - xb) * (probs - xb))
        
        
        #X,pre_act,post_act = params['cache_output']
        #delta1 = (probs-yb)*sigmoid_deriv(pre_act)*(1-sigmoid_deriv(pre_act))
        delta1 = 2.0 *(probs-xb)
        delta2 = backwards(delta1,params,'output',sigmoid_deriv)
        delta3 = backwards(delta2,params,'layer3',relu_deriv)
        delta4 = backwards(delta3,params,'layer2',relu_deriv)
        backwards(delta4,params,'layer1',relu_deriv)
        
        W_layer1 = params['W' + 'layer1']        
        b_layer1 = params['b' + 'layer1']
        W_layer2 = params['W' + 'layer2']        
        b_layer2 = params['b' + 'layer2']
        W_layer3 = params['W' + 'layer3']        
        b_layer3 = params['b' + 'layer3']
        W_output = params['W' + 'output']        
        b_output = params['b' + 'output']
        MW_layer1= params["Momentum"+"W"+"layer1"] 
        MW_layer2=params["Momentum"+"W"+"layer2"] 
        MW_layer3=params["Momentum"+"W"+"layer3"] 
        MW_output=params["Momentum"+"W"+"output"] 
        Mb_layer1=params["Momentum"+"b"+"layer1"] 
        Mb_layer2=params["Momentum"+"b"+"layer2"] 
        Mb_layer3=params["Momentum"+"b"+"layer3"]
        Mb_output=params["Momentum"+"b"+"output"] 
        grad_W_layer1 = params['grad_W' + 'layer1']        
        grad_b_layer1 = params['grad_b' + 'layer1']        
        grad_W_layer2 = params['grad_W' + 'layer2']        
        grad_b_layer2 = params['grad_b' + 'layer2']  
        grad_W_layer3 = params['grad_W' + 'layer3']        
        grad_b_layer3 = params['grad_b' + 'layer3']
        grad_W_output = params['grad_W' + 'output']        
        grad_b_output = params['grad_b' + 'output']
        MW_layer1 = 0.9*MW_layer1 - learning_rate*grad_W_layer1        
        Mb_layer1 = 0.9*Mb_layer1 - learning_rate*grad_b_layer1
        W_layer1 = W_layer1 + MW_layer1        
        b_layer1 = b_layer1 + Mb_layer1
        MW_layer2 = 0.9*MW_layer2 - learning_rate*grad_W_layer2        
        Mb_layer2 = 0.9*Mb_layer2 - learning_rate*grad_b_layer2
        W_layer2 = W_layer2 + MW_layer2        
        b_layer2 = b_layer2 + Mb_layer2
        MW_layer3 = 0.9*MW_layer3 - learning_rate*grad_W_layer3        
        Mb_layer3 = 0.9*Mb_layer3 - learning_rate*grad_b_layer3
        W_layer3 = W_layer3 + MW_layer3        
        b_layer3 = b_layer3 + Mb_layer3        
        MW_output = 0.9*MW_output - learning_rate*grad_W_output        
        Mb_output = 0.9*Mb_output - learning_rate*grad_b_output   
        W_output = W_output + MW_output        
        b_output = b_output + Mb_output
        
        
        params['W' + 'layer1']=W_layer1        
        params['b' + 'layer1']=b_layer1
        params['W' + 'layer2']=W_layer2        
        params['b' + 'layer2']=b_layer2
        params['W' + 'layer3']=W_layer3        
        params['b' + 'layer3']=b_layer3
        params['W' + 'output']=W_output        
        params['b' + 'output']=b_output
        params["Momentum"+"W"+"layer1"]=MW_layer1 
        params["Momentum"+"W"+"layer2"]=MW_layer2 
        params["Momentum"+"W"+"layer3"]=MW_layer3 
        params["Momentum"+"W"+"output"]=MW_output 
        params["Momentum"+"b"+"layer1"]=Mb_layer1 
        params["Momentum"+"b"+"layer2"]=Mb_layer2 
        params["Momentum"+"b"+"layer3"]=Mb_layer3
        params["Momentum"+"b"+"output"]=Mb_output
        
        
        
        total_loss +=loss
        
    losses_list.append(total_loss)
        

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
        
# Q5.3.1
import matplotlib.pyplot as plt
# visualize some results
##########################
##### your code here #####
##########################
#print(len(losses_list))
plt.figure(0)
epoch_list = np.arange(max_iters)
plt.title('Training Loss for Auto encoder')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(losses_list,epoch_list)
plt.show()

j=0
for i in range(10):
    
    h1 = forward(valid_x[j,:],params,'layer1',relu)
    h2 = forward(h1,params,'layer2',relu)
    h3 = forward(h2,params,'layer3',relu)
    probs = forward(h3,params,'output',sigmoid)
    
    

    
    
    probs_reshaped = probs.reshape(32,32).T
    valid_reshaped = valid_x[j,:].reshape(32,32).T
    plt.imshow(valid_reshaped)
    plt.show()
    plt.imshow(probs_reshaped)
    plt.show()
    if(i%2==0):
        j+=1
    else:
        j+=100


# Q5.3.2
from skimage.measure import compare_psnr as psnr
# evaluate PSNR
##########################
##### your code here #####
##########################
psnr_val=0
for i in range(valid_x.shape[0]):
    h1 = forward(valid_x[i,:],params,'layer1',relu)
    h2 = forward(h1,params,'layer2',relu)
    h3 = forward(h2,params,'layer3',relu)
    probs = forward(h3,params,'output',sigmoid)
    psnr_val += psnr(valid_x[i,:],probs)

print("Average psnr value", psnr_val/valid_x.shape[0])
    
