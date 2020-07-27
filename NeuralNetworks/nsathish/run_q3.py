import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']


max_iters = 100
# pick a batch size, learning rate
batch_size = None
learning_rate = None
hidden_size = 64
##########################
##### your code here #####
##########################
#print(train_x.shape) #10800*1023
#print(valid_x.shape) #3600
#print(train_y.shape) #10800*36

batch_size = 32
learning_rate = 3e-3

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
##########################
##### your code here #####
##########################

initialize_weights(1024,64,params,'layer1')
initialize_weights(64,36,params,'output')

TrainLoss=[]
TrainAcc=[]
ValidLoss=[]
ValidAcc=[]
# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # training loop can be exactly the same as q2!
        ##########################
        ##### your code here #####
        ##########################
        h1 = forward(xb,params,'layer1')
        probs = forward(h1,params,'output',softmax)        
        loss, acc = compute_loss_and_acc(yb, probs)        
        delta1 = probs        
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
        total_acc += acc
    TrainLoss.append(total_loss)
    
    total_acc = total_acc/batch_num
    TrainAcc.append(total_acc)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
    
    h1_valid = forward(valid_x, params, 'layer1')
    probs_valid = forward(h1_valid , params, 'output',softmax)
    valid_loss, valid_acc = compute_loss_and_acc(valid_y,probs_valid)
    ValidLoss.append(valid_loss)
    ValidAcc.append(valid_acc)
    
    
    print("Valid acc", valid_acc)

# run on validation set and report accuracy! should be above 75%
#valid_acc = None
##########################
##### your code here #####
##########################
plt.figure(1)
plt.title('Training Loss and test loss')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(TrainLoss , 'r')
plt.plot(ValidLoss, 'b')
plt.legend(['training loss','test loss'])
plt.show()


# In[61]:


plt.figure(0)
plt.title('Training ACC and test ACC')
plt.xlabel('Epoch Number')
plt.ylabel('ACC')
plt.plot(TrainAcc , 'r')
plt.plot(ValidAcc, 'b')
plt.legend(['training ACC','test ACC'])

plt.show()
    

print('Validation accuracy: ',valid_acc)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# visualize weights here
##########################
##### your code here #####
##########################
file = open('q3_weights.pickle', 'rb')
weights = pickle.load(file)
weights_layer1 = weights['Wlayer1']

fig = plt.figure(3)
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )
for i in range(weights_layer1.shape[1]):
    weights = weights_layer1[:,i].reshape(32,32)
    grid[i].imshow(weights)

plt.show()

initialize_weights(1024, 64, params, 'orig')
weights_orig = params['Worig']

fig = plt.figure(4)
grid = ImageGrid(fig, 111,nrows_ncols=(8,8), axes_pad=0.1)
for i in range(weights_orig.shape[1]):
    weight_i = weights_orig[:,i].reshape(32, 32)
    grid[i].imshow(weight_i)
plt.show()
    
    

# Q3.1.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))






# compute comfusion matrix here
##########################
##### your code here #####
##########################
h1 = forward(test_x, params, 'layer1')
probs = forward(h1, params, 'output', softmax)

labels= np.argmax(test_y, axis=1)
predicted = np.argmax(probs, axis=1)



for i in range(labels.shape[0]):
    confusion_matrix[labels[i], predicted[i]] += 1



import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()