import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
#from mmd_util import rb
from sklearn import metrics

# Generate toy data that has two distinct classes and a huge gap between them
X, Y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=0.4)  # X - features, Y - labels
Y[Y == 0] = -1  # Replace zeros with -1
# Plot the toy data
#plt.scatter(x=X[:, 0], y=X[:, 1])


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

def rbf_kernel(X, gamma=None):
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    n = X.shape[0]
    K = Variable(torch.zeros(n,n), requires_grad=False)
    for i in range(n):
        for j in range(n):
            K[i][j] = torch.dist(X[i],X[j])**2
    #K = euclidean_distances(X, Y, squared=True)
    K1 = K*-gamma
    K2 = K1*torch.exp(K)  # exponentiate K in-place
    return K2

class SVM(nn.Module):
    def __init__(self, input_dimension):
        super().__init__()  # Call the init function of nn.Module
        self.fully_connected = nn.Linear(input_dimension, 1)  # Implement the Linear function
        
    def forward(self, x):
        #x = (x - x.mean())/x.std()  # Feature scaling
        #K = rbf_kernel(x)
        fwd = self.fully_connected(x)  # Forward pass
        return fwd


learning_rate = 0.01  # Learning rate
epochs = 100  # Number of epochs

x = torch.FloatTensor(X)
y = torch.FloatTensor(Y)
N = X.shape[1]  # Number of samples, 500

model = SVM(N)  # Our model

for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
        
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # Our optimizer
model.train()  # Our model, SVM is a subclass of the nn.Module, so it inherits the train method
for epoch in range(epochs):
    
    #x = Variable(X)  # Convert features and classes to variables
    #y = Variable(Y)

    optimizer.zero_grad()  # Manually zero the gradient buffers of the optimizer
    output = model(x)  # Compute the output by doing a forward pass
    output = torch.flatten(output)

    loss = torch.mean(torch.clamp(1 - output * y, min=0))  # hinge loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Optimize and adjust weights

    #sum_loss += loss.data.cpu().numpy()  # Add the loss
    
    print("Epoch {}, Loss: {}".format(epoch, loss))


model.eval()

correct = 0

for i in range(N):
    x1 = x[i]
    y1 = y[i]

    if model(x1)*y1 >= 1:
        correct += 1

print(correct)
