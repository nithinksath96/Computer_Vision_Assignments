import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

test_data = scipy.io.loadmat('../data/nist36_test.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

test_x  = test_data['test_data']

dim = 32
# do PCA
##########################
##### your code here #####
##########################
x_mean  = np.mean(train_x)
x_std = train_x - x_mean
x_square = np.matmul(x_std.T,x_std)
#print(x_square.shape)
u,s,vt= np.linalg.svd(x_square)
#print(s)
#print(u.shape)
proj_matrix = u[:,:dim]


j=0

valid_std = valid_x - np.mean(valid_x)

test_std = test_x - np.mean(test_x)

transformed_test = np.matmul(test_std , proj_matrix)
reprojected_test = np.matmul(transformed_test, np.transpose(proj_matrix))

for i in range(10):
    

    
    
    transfomed_valid = np.matmul(valid_std[j,:].reshape(1,1024), proj_matrix)
    print(transfomed_valid.shape)
    reprojected_valid = np.matmul(transfomed_valid, np.transpose(proj_matrix))
    print(reprojected_valid.shape)
    

    reprojected_valid = reprojected_valid.reshape(32,32).T
    valid_reshaped = valid_std[j,:].reshape(32,32).T
    plt.imshow(valid_reshaped)
    plt.show()
    plt.imshow(reprojected_valid)
    plt.show()
    if(i%2==0):
        j+=1
    else:
        j+=100

transformed_valid = np.matmul(valid_std , proj_matrix)
reprojected_valid = np.matmul(transformed_valid, np.transpose(proj_matrix))
reprojected_valid+=np.mean(valid_x)
psnr_val =0
for i in range(valid_x.shape[0]):
    psnr_val+=psnr(valid_x[i,:].flatten(), reprojected_valid[i,:].flatten())

print(psnr_val/valid_x.shape[0])

    #break
#print(proj_matrix.shape)
#inv_proj = np.linalg.inv(proj_matrix)
#print(inv_proj.shape)
    

# rebuild a low-rank version
lrank = None
##########################
##### your code here #####
##########################

# rebuild it
recon = None
##########################
##### your code here #####
##########################

# build valid dataset
recon_valid = None
##########################
##### your code here #####
##########################

# visualize the comparison and compute PSNR
##########################
##### your code here #####
##########################
