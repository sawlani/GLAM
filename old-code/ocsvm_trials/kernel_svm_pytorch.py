import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
#from mmd_util import rb
from sklearn import metrics

# Generate toy data that has two distinct classes and a huge gap between them
X, Y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=0.4)  # X - features, Y - labels
# Plot the toy data
plt.scatter(x=X[:, 0], y=X[:, 1])


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

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
        self.fully_connected = nn.Linear(input_dimension, 1)  # Implement the Linear function
        
    def forward(self, x):
        fwd = self.fully_connected(x)  # Forward pass
        return fwd


data = X  # Before feature scaling
X = (X - X.mean())/X.std()  # Feature scaling
K = metrics.pairwise.rbf_kernel(X,X)
Y[Y == 0] = -1  # Replace zeros with -1
#plt.scatter(x=X[:, 0], y=X[:, 1])  # After feature scaling
#plt.scatter(x=data[:, 0], y=data[:, 1], c='r')  # Before feature scaling



learning_rate = 0.01  # Learning rate
epoch = 10  # Number of epochs
batch_size = 500  # Batch size

K = torch.FloatTensor(K)  # Convert X and Y to FloatTensors
Y = torch.FloatTensor(Y)
N = len(Y)  # Number of samples, 500

'''
K = torch.zeros(N,N)
gamma = 1/N

for i in range(N):
    for j in range(N):
        K[i,j] = torch.exp(-gamma * (torch.dist(X[i],X[j]))**2)
'''

model = SVM(N)  # Our model
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # Our optimizer
model.train()  # Our model, SVM is a subclass of the nn.Module, so it inherits the train method
for epoch in range(epoch):
    perm = torch.randperm(N)  # Generate a set of random numbers of length: sample size
    sum_loss = 0  # Loss for each epoch
        
    for i in range(0, N, batch_size):
        k = K[perm[i:i + batch_size]]  # Pick random samples by iterating over random permutation
        y = Y[perm[i:i + batch_size]]  # Pick the correlating class
        
        k = Variable(k)  # Convert features and classes to variables
        y = Variable(y)

        optimizer.zero_grad()  # Manually zero the gradient buffers of the optimizer
        output = model(k)  # Compute the output by doing a forward pass
        
        output = torch.flatten(output)
        #print(output.shape)
        #print(y.shape)
        loss = torch.mean(torch.clamp(1 - output * y, min=0))  # hinge loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimize and adjust weights

        sum_loss += loss.data.cpu().numpy()  # Add the loss
        
    print("Epoch {}, Loss: {}".format(epoch, sum_loss))

model.eval()

correct = 0

for i in range(N):
    k = K[i]
    y = Y[i]

    if model(k)*y >= 1:
        correct += 1

print(correct)