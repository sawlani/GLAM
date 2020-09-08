import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class SVM(nn.Module):
    
    def __init__(self, input_dimension, output_dimension):
        super().__init__()  # Call the init function of nn.Module
        self.fully_connected = nn.Linear(input_dimension, output_dimension)  # Implement the Linear function
        
    def forward(self, X):
        fwd = self.fully_connected(X)  # Forward pass
        return fwd


def find_center(model, X):
    model.eval()  # Our model, SVM is a subclass of the nn.Module, so it inherits the train method
        
    output = model(X)  # Compute the output by doing a forward pass
    c = torch.mean(output, dim=0)

    return c

    
def train(model, optimizer, c, X):    
    model.train()  # Our model, SVM is a subclass of the nn.Module, so it inherits the train method
        
    optimizer.zero_grad()  # Manually zero the gradient buffers of the optimizer
    output = model(X)  # Compute the output by doing a forward pass
    
    #weights = model.fully_connected.weight.squeeze()
    #weights = torch.matmul(weights, K)
    
    dist = torch.sum((output - c) ** 2, dim=1)
    loss = torch.mean(dist)
    
    loss.backward()  # Backpropagation
    optimizer.step()  # Optimize and adjust weights

    loss_value = loss.data.cpu().numpy()  # Add the loss
    
    return loss_value

def test(model, c, X, Y):

    model.eval()
    
    correct_normal = 0
    correct_outliers = 0
    output = model(X)  # Compute the output by doing a forward pass
    
    dist = torch.sum((output - c) ** 2, dim=1)
    dist = dist.detach().numpy()


    q = np.quantile(dist, 0.95)
    for i in range(len(Y)):
        if dist[i] >= q and Y[i] == -1:
            correct_outliers += 1
        elif dist[i] < q and Y[i] == 1:
            correct_normal += 1

    print(correct_outliers)
    print(correct_normal)
    #print("accuracy=%f" % (correct/len(Y)))


# Generate toy data that has two distinct classes and a huge gap between them
X, Y = make_blobs(n_samples=[5,95], centers=None, random_state=0, cluster_std=0.4)  # X - features, Y - labels
Y[Y == 0] = -1  # Replace zeros with -1

learning_rate = 0.01  # Learning rate
epochs= 100  # Number of epochs

X = torch.FloatTensor(X)  # Convert X and Y to FloatTensors
Y = torch.FloatTensor(Y)
N = len(Y)  # Number of samples, 500

dim = len(X[0])
model = SVM(dim, 2)  # Our model
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)  # Our optimizer

plt.scatter(x=X[:, 0], y=X[:, 1])
plt.show()
#X = (X - X.mean())/X.std()
#K = rbf_kernel(X)
for epoch in range(epochs):
    if epochs % 10 == 0:
        c = find_center(model, X)
    loss_value = train(model, optimizer, c, X)
    print("Epoch {}, Loss: {}".format(epoch, loss_value))
    test(model, c, X, Y)
