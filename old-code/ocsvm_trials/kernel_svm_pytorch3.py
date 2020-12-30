import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
import os
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
    K1 = K*-gamma
    K2 = K1*torch.exp(K)  # exponentiate K in-place
    return K2



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
        self.trial_layer = nn.Linear(2,2)
        self.fully_connected = nn.Linear(input_dimension, 1)  # Implement the Linear function
        
    def forward(self, XX):
        X = self.trial_layer(XX)
        X = (X - X.mean())/X.std()  # Feature scaling
        K = rbf_kernel(X)

        pred = self.fully_connected(K)  # Forward pass
        fwd = torch.flatten(pred)
        return fwd


#data = X  # Before feature scaling
#plt.scatter(x=X[:, 0], y=X[:, 1])  # After feature scaling
#plt.scatter(x=data[:, 0], y=data[:, 1], c='r')  # Before feature scaling



def train(model, epochs, optimizer, X, Y):    
    model.train()  # Our model, SVM is a subclass of the nn.Module, so it inherits the train method

    for epoch in range(epochs):
            
        optimizer.zero_grad()  # Manually zero the gradient buffers of the optimizer
        output = model(X)  # Compute the output by doing a forward pass
        
        
        loss = torch.mean(torch.clamp(1 - output * Y, min=0))  # hinge loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimize and adjust weights

        loss_value = loss.data.cpu().numpy()  # Add the loss
        
        print("Epoch {}, Loss: {}".format(epoch, loss_value))

def test(model, X, Y):

    model.eval()

    output = model(X)  # Compute the output by doing a forward pass
    correct = torch.sum(output*Y >= 1)

    print(correct)


# Generate toy data that has two distinct classes and a huge gap between them
X, Y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=0.4)  # X - features, Y - labels
Y[Y == 0] = -1  # Replace zeros with -1

learning_rate = 0.01  # Learning rate
epochs= 3  # Number of epochs

X = torch.FloatTensor(X)  # Convert X and Y to FloatTensors
Y = torch.FloatTensor(Y)
N = len(Y)  # Number of samples, 500


model = SVM(N)  # Our model
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # Our optimizer




# Plot the toy data
#plt.scatter(x=X[:, 0], y=X[:, 1])

train(model, epochs, optimizer, X, Y)
test(model, X, Y)