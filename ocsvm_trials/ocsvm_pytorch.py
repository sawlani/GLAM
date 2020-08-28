import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.svm import OneClassSVM
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def rbf_kernel(X, gamma=None):
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    n = X.shape[0]
    K = torch.zeros(n,n)
    for i in range(n):
        for j in range(n):
            K[i][j] = torch.dist(X[i],X[j])**2
    #K = euclidean_distances(X, Y, squared=True)
    K = K*-gamma
    K = K*torch.exp(K)  # exponentiate K in-place
    return K



class SVM(nn.Module):
    """
    Linear Support Vector Machine
    -----------------------------
    This SVM is a subclass of the PyTorch nn module that
    implements the Linear  function. The  size  of  each 
    input sample is 2 and output sample  is 1.
    """
    def __init__(self, input_dimension):
        super().__init__()  # Call the init function of nn.Module
        #self.trial_layer = nn.Linear(2,2)
        self.fully_connected = nn.Linear(input_dimension, 1)  # Implement the Linear function
        
    def forward(self, X):
        #X = self.trial_layer(XX)
          # Feature scaling
        

        fwd = self.fully_connected(K)  # Forward pass
        #fwd = torch.flatten(fwd)
        return fwd


#data = X  # Before feature scaling
#plt.scatter(x=X[:, 0], y=X[:, 1])  # After feature scaling
#plt.scatter(x=data[:, 0], y=data[:, 1], c='r')  # Before feature scaling



def train(model, rho, optimizer, K, nu = 0.2):    
    model.train()  # Our model, SVM is a subclass of the nn.Module, so it inherits the train method

    
        
    optimizer.zero_grad()  # Manually zero the gradient buffers of the optimizer
    output = model(K)  # Compute the output by doing a forward pass
    weights = model.fully_connected.weight.squeeze()
    weights = torch.matmul(weights, K)
    
    
    loss1 = (1/nu)*torch.mean(torch.clamp(rho - output, min=0))  # hinge loss
    loss2 = torch.dot(weights,weights)
    print("rho=%f" % rho)
#    print(loss.shape)
#    print(rho.shape)
    loss3 = -rho.squeeze()


    print(loss1.data.numpy(),loss2.data.numpy(),loss3.data.numpy())
    loss = loss1 + loss2 + loss3

    loss.backward()  # Backpropagation
    optimizer.step()  # Optimize and adjust weights

    loss_value = loss.data.cpu().numpy()  # Add the loss
    
    return loss_value

def test(model, K, Y, rho):

    model.eval()
    rho_ = rho.detach()
    correct = 0
    output = model(K)  # Compute the output by doing a forward pass
    for i in range(len(Y)):
    	if Y[i]*(output[i]-rho) >= 0:
    		correct += 1
    #correct = torch.sum( torch.Tensor(Y) == torch.sign(output - rho))

    #print(output)
    #print(Y)
    print("accuracy=%f" % (correct/len(Y)))


# Generate toy data that has two distinct classes and a huge gap between them
X, Y = make_blobs(n_samples=[5,95], centers=None, random_state=0, cluster_std=0.4)  # X - features, Y - labels
Y[Y == 0] = -1  # Replace zeros with -1
#Y[Y == 0] = 1  # Replace zeros with -1
#training_Y = 1*len(Y)
#print(sum(Y))

learning_rate = 0.01  # Learning rate
epochs= 500  # Number of epochs

X = torch.FloatTensor(X)  # Convert X and Y to FloatTensors
Y = torch.FloatTensor(Y)
N = len(Y)  # Number of samples, 500


model = SVM(N)  # Our model
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # Our optimizer

rho = torch.randn(1)
rho.requires_grad = True

optimizer.add_param_group({'params': rho})
'''
for nu in np.arange(0.05, 0.55, 0.05):
	baseline_labels = OneClassSVM(nu=nu, gamma = 'scale').fit_predict(X)
	correct_outliers = 0
	correct_regular = 0
	for i in range(len(X)):
		#print(i, X[i], baseline_labels[i], Y[i])
		if baseline_labels[i] == Y[i]:
			if Y[i] == -1:
				correct_outliers += 1
			else:
				correct_regular += 1

	print(nu, correct_regular, correct_outliers)


# Plot the toy data
'''
plt.scatter(x=X[:, 0], y=X[:, 1])
plt.show()
X = (X - X.mean())/X.std()
K = rbf_kernel(X)
for epoch in range(epochs):
	loss_value = train(model, rho, optimizer, K)
	print("Epoch {}, Loss: {}".format(epoch, loss_value))
	test(model, K, Y, rho)
