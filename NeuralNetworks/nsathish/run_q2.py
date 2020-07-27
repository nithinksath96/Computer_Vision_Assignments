import numpy as np
# you should write your functions in nn.py
from nn import *
from util import *




# fake data
# feel free to plot it in 2D
# what do you think these 4 classes are?
g0 = np.random.multivariate_normal([3.6,40],[[0.05,0],[0,10]],10)
g1 = np.random.multivariate_normal([3.9,10],[[0.01,0],[0,5]],10)
g2 = np.random.multivariate_normal([3.4,30],[[0.25,0],[0,5]],10)
g3 = np.random.multivariate_normal([2.0,10],[[0.5,0],[0,10]],10)
x = np.vstack([g0,g1,g2,g3])
# we will do XW + B
# that implies that the data is N x D

# create labels
y_idx = np.array([0 for _ in range(10)] + [1 for _ in range(10)] + [2 for _ in range(10)] + [3 for _ in range(10)])
# turn to one_hot
y = np.zeros((y_idx.shape[0],y_idx.max()+1))
y[np.arange(y_idx.shape[0]),y_idx] = 1

# parameters in a dictionary
params = {}


# Q 2.1
# initialize a layer
initialize_weights(2,25,params,'layer1')
initialize_weights(25,4,params,'output')
assert(params['Wlayer1'].shape == (2,25))
assert(params['blayer1'].shape == (25,))



#expect 0, [0.05 to 0.12]
print("{}, {:.2f}".format(params['blayer1'].sum(),params['Wlayer1'].std()**2))
print("{}, {:.2f}".format(params['boutput'].sum(),params['Woutput'].std()**2))

# Q 2.2.1
# implement sigmoid
test = sigmoid(np.array([-1000,1000]))
print('should be zero and one\t',test.min(),test.max())
# implement forward
h1 = forward(x,params,'layer1')
#print(x.shape) #(40,2)
print(h1.shape) #(40,25)
# Q 2.2.2
# implement softmax
probs = forward(h1,params,'output',softmax)
# make sure you understand these values!
# positive, ~1, ~1, (40,4)
print(probs.min(),min(probs.sum(1)),max(probs.sum(1)),probs.shape)

# Q 2.2.3
# implement compute_loss_and_acc
loss, acc = compute_loss_and_acc(y, probs)
# should be around -np.log(0.25)*40 [~55] and 0.25
# if it is not, check softmax!
print("{}, {:.2f}".format(loss,acc))

# here we cheat for you
# the derivative of cross-entropy(softmax(x)) is probs - 1[correct actions]
delta1 = probs
delta1[np.arange(probs.shape[0]),y_idx] -= 1

# we already did derivative through softmax
# so we pass in a linear_deriv, which is just a vector of ones
# to make this a no-op
delta2 = backwards(delta1,params,'output',linear_deriv)
# Implement backwards!
backwards(delta2,params,'layer1',sigmoid_deriv)

# W and b should match their gradients sizes
for k,v in sorted(list(params.items())):
    if 'grad' in k:
        name = k.split('_')[1]
        print(name,v.shape, params[name].shape)

# Q 2.4
batches = get_random_batches(x,y,5)
# print batch sizes
print([_[0].shape[0] for _ in batches])
batch_num = len(batches)
#print("batch number",batch_num)
# WRITE A TRAINING LOOP HERE
max_iters = 500
learning_rate = 1e-3
# with default settings, you should get loss < 35 and accuracy > 75%
for itr in range(max_iters):
    total_loss = 0
    avg_acc = 0
    for xb,yb in batches:
        #pass
        # forward
        

        # loss
        # be sure to add loss and accuracy to epoch totals 

        # backward

        # apply gradient

        ##########################
        ##### your code here #####
        ##########################
        h1 = forward(xb,params,'layer1')
        
        
        probs = forward(h1,params,'output',softmax)
        
        loss, acc = compute_loss_and_acc(yb, probs)
        
        #print("Accuraccy",acc)
        
        delta1 = probs
        
#        print("delta1",delta1.shape)
#        
#        print("Probs",probs.shape)
#        
#        print("Probs shape[0]",probs.shape[0])
        
        delta1 = probs - yb
        
        delta2 = backwards(delta1,params,'output',linear_deriv)
        
        backwards(delta2,params,'layer1',sigmoid_deriv)
        
        W_layer1 = params['W' + 'layer1']
        
        b_layer1 = params['b' + 'layer1']
        
        W_output = params['W' + 'output']
        
        b_output = params['b' + 'output']
        
        grad_W_layer1 = params['grad_W' + 'layer1']
        
        grad_b_layer1 = params['grad_b' + 'layer1']
        
        grad_W_output = params['grad_W' + 'output']
        
        grad_b_output = params['grad_b' + 'output']
        
        W_layer1 = W_layer1 - learning_rate*grad_W_layer1
        
        b_layer1 = b_layer1 - learning_rate*grad_b_layer1
        
        W_output = W_output - learning_rate*grad_W_output
        
        b_output = b_output - learning_rate*grad_b_output
        
        params['W' + 'layer1'] =  W_layer1 
        
        params['b' + 'layer1'] = b_layer1
        
        params['W' + 'output'] = W_output
        
        params['b' + 'output'] = b_output
        
        
        
        total_loss+= loss
        
        avg_acc += acc
        
        #print("Avg acc",avg_acc)
        
        
        
        
        
        
        

    avg_acc = avg_acc / batch_num
    if itr % 100 == 0:
        #avg_acc = avg_acc /100
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))


# Q 2.5 should be implemented in this file
# you can do this before or after training the network. 

##########################
##### your code here #####
##########################

# save the old params
import copy
params_orig = copy.deepcopy(params)

params1= copy.deepcopy(params)

eps = 1e-6

xb,yb = batches[-1]
for k,v in params.items():
    #print("K",k)
    #print("V",v)
    if '_' in k: 
        continue
    # we have a real parameter!
    # for each value inside the parameter
    #   add epsilon
    #   run the network
    #   get the loss
    #   compute derivative with central diffs
    
    ##########################
    ##### your code here #####
    ##########################
    elif('Wlayer1' in k):
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                
                original= v[i,j]
                
                params1['Wlayer1'][i,j] = original + eps
                h1 = forward(xb,params1,'layer1')
                probs = forward(h1,params1,'output',softmax)
                loss_eps1, acc = compute_loss_and_acc(yb, probs)
                
                params1['Wlayer1'][i,j] = original - eps
                h1 = forward(xb,params1,'layer1')
                probs = forward(h1,params1,'output',softmax)
                loss_eps2, acc = compute_loss_and_acc(yb, probs)
                
                params1['Wlayer1'][i,j] = original
                params_orig['grad_W' + 'layer1'][i,j] = (loss_eps1 - loss_eps2) /(2*eps)
                
                

        
    elif('blayer1' in k):
        
        for i in range(v.shape[0]):
            
            original = v[i]
            params1['blayer1'][i] = original + eps
            h1 = forward(xb,params1,'layer1')
            probs = forward(h1,params1,'output',softmax)
            loss_eps1, acc = compute_loss_and_acc(yb, probs)
            
            params1['blayer1'][i] = original - eps
            h1 = forward(xb,params1,'layer1')
            probs = forward(h1,params1,'output',softmax)
            loss_eps2, acc = compute_loss_and_acc(yb, probs)
            
            params1['blayer1'][i] = original
            params_orig['grad_b' + 'layer1'][i] = (loss_eps1 - loss_eps2) /(2*eps)
        
                

    elif('Woutput' in k):
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                
                original= v[i,j]
                
                params1['Woutput'][i,j] = original + eps
                h1 = forward(xb,params1,'layer1')
                probs = forward(h1,params1,'output',softmax)
                loss_eps1, acc = compute_loss_and_acc(yb, probs)
                
                params1['Woutput'][i,j] = original - eps
                h1 = forward(xb,params1,'layer1')
                probs = forward(h1,params1,'output',softmax)
                loss_eps2, acc = compute_loss_and_acc(yb, probs)
                
                params1['Woutput'][i,j] = original
                params_orig['grad_W' + 'output'][i,j] = (loss_eps1 - loss_eps2) /(2*eps)
        

    elif('boutput' in k):
        for i in range(v.shape[0]):
            
            original = v[i]
            params1['boutput'][i] = original + eps
            h1 = forward(xb,params1,'layer1')
            probs = forward(h1,params1,'output',softmax)
            loss_eps1, acc = compute_loss_and_acc(yb, probs)
            
            params1['boutput'][i] = original - eps
            h1 = forward(xb,params1,'layer1')
            probs = forward(h1,params1,'output',softmax)
            loss_eps2, acc = compute_loss_and_acc(yb, probs)
            
            params1['boutput'][i] = original
            params_orig['grad_b' + 'output'][i] = (loss_eps1 - loss_eps2) /(2*eps)



        
#    W_layer1 = params['W' + 'layer1']
#        
#    b_layer1 = params['b' + 'layer1']
#        
#    W_output = params['W' + 'output']
#        
#    b_output = params['b' + 'output']
#    
    

total_error = 0
for k in params.keys():
    if 'grad_' in k:
        # relative error
        #print("--------------------------- ",k,params[k])
        #print("+++++++++++++++++++++++++++ ",k,params_orig[k])
        err = np.abs(params[k] - params_orig[k])/np.maximum(np.abs(params[k]),np.abs(params_orig[k]))
        err = err.sum()
        print('{} {:.2e}'.format(k, err))
        total_error += err
# should be less than 1e-4
print('total {:.2e}'.format(total_error))
