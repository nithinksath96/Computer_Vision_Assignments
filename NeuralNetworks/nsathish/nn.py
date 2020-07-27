import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None

    ##########################
    ##### your code here #####
    ##########################
    
    Xavier = np.sqrt(6) / np.sqrt(in_size + out_size)
    
    W = np.random.uniform(-Xavier,Xavier,in_size*out_size).reshape(in_size,out_size)
    b= np.zeros(out_size)
    
    params['W' + name] = W
    params['b' + name] = b
    

    
    

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = None

    ##########################
    ##### your code here #####
    ##########################
    
    res = 1.0 /(1.0 + np.exp(-x))

    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]


    ##########################
    ##### your code here #####
    ##########################
    
    pre_act= np.matmul(X,W) + b
    
    post_act = activation(pre_act)


    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None

    ##########################
    ##### your code here #####
    ##########################
    
    c=np.max(x,axis=1).reshape(-1,1)
    d=x-c
    exp_sum=np.sum(np.exp(d),axis=1).reshape(-1,1)
    #exp_sum= np.reshape(exp_sum, (exp_sum.shape[0],1))
    res= d-np.log(exp_sum)
    
    res = np.exp(res)
    return res
    

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    ##########################
    ##### your code here #####
    ##########################
    
    loss = -1.0*np.sum(y*np.log(probs))
    
    temp =  np.where(np.argmax(y,axis=1)==np.argmax(probs,axis=1),1,0)
    
    acc = float(np.sum(temp)/y.shape[0])
    
    #acc= np.sum(np.equal(np.argmax(y, axis=-1), np.argmax(probs, axis=-1))) / y.shape[0]
    
    #print("Total currect",np.sum(temp)) #3
    #print("Out of total",y.shape[0]) #5
    #print("Accuracy",acc)

    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # then compute the derivative W,b, and X
    ##########################
    ##### your code here #####
    ##########################
    #dz= np.multiply(dy, self.activations[i].derivative())
    #dy=np.matmul(dz,np.transpose(self.W[i]))
    #self.dW[i]=np.matmul(np.transpose(self.dzw[i]),dz)
    #self.db[i]=np.sum(dz,axis=0)
    #print("Delta shape",delta.shape)
    #print("Activation derivative", activation_deriv(post_act).shape)
    #print("Pre act",pre_act.shape)
    #print("Post act",post_act.shape)
    #print("Input",X.shape)
    #print("----------------------W shape",W.shape)
    #delta = delta/X.shape[0]
    dz = np.multiply(delta , activation_deriv(post_act))
    
    grad_X = np.matmul(dz, np.transpose(W))
    grad_W = np.matmul(np.transpose(X), dz)
    grad_b = np.sum(dz,axis=0)
    #grad_b = np.dot(np.ones((1, delta.shape[0])), delta).reshape(-1)
    #print("------------------------DW shape", grad_W.shape)
    #print("--------------------------DX shape", grad_X.shape)

    

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X


############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    ##########################
    ##### your code here #####
    ##########################

    temp = np.arange(x.shape[0])
    
    temp = np.random.permutation(temp)

    nbatches = int(np.ceil(x.shape[0]/batch_size))
    
    
    j=0
    for i in range(nbatches):
        
        batches.append((x[temp[j:j+batch_size]],y[temp[j:j+batch_size]]))
        j+=batch_size
#    print("----------------------------------------------------X",x.shape)
#    print("----------------------------------------------------Y",y.shape)
    return batches

