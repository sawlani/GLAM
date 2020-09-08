import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    
def train(radius, alphas, optimizer, K):    
    #model.train()  # Our model, SVM is a subclass of the nn.Module, so it inherits the train method
        
    optimizer.zero_grad()  # Manually zero the gradient buffers of the optimizer
    
    normalized_alphas = alphas/torch.sum(alphas)

    alpha_matrix = torch.ger(normalized_alphas, normalized_alphas)

    #dim = K.shape[0]
    diag = torch.diag(K)
    row_dots = torch.matmul(K, normalized_alphas)
    total_dot = torch.dot(torch.flatten(alpha_matrix), torch.flatten(K))
    
    dists = diag - 2*row_dots + total_dot
    scores = torch.clamp(dists - (radius**2), min=0)
    loss = torch.mean(scores)

    loss.backward()  # Backpropagation
    optimizer.step()  # Optimize and adjust weights

    loss_value = loss.data.cpu().numpy()  # Add the loss
    
    return loss_value

def test(radius, alphas, K, Y):
    #print("radius="+str(radius))
    
    with torch.no_grad():

        normalized_alphas = alphas/torch.sum(alphas)

        alpha_matrix = torch.ger(normalized_alphas, normalized_alphas)

        #dim = K.shape[0]
        diag = torch.diag(K)
        row_dots = torch.matmul(K, normalized_alphas)
        total_dot = torch.dot(torch.flatten(alpha_matrix), torch.flatten(K))

        dists = diag - 2*row_dots + total_dot

        preds = (dists <= (radius**2))

        p, r, f, _ = precision_recall_fscore_support(Y, preds, average="binary")
        
        return p,r,f


def run_experiment(X,Y, learning_rate=0.01, epochs=500, weight_decay = 10, learn_center=False):

    K = rbf_kernel(X)

    K = torch.FloatTensor(K)  # Convert K and Y to FloatTensors
    Y = torch.FloatTensor(Y)
    num_samples = len(Y)

    radius = torch.tensor(0.1, requires_grad=True)
    if learn_center:
        alphas = torch.tensor([1.]*num_samples, requires_grad=True)
        optimizer = optim.Adam([radius,alphas], lr=learning_rate, weight_decay=weight_decay)  # Our optimizer
    else:
        alphas = torch.tensor([1.]*num_samples)
        optimizer = optim.Adam([radius], lr=learning_rate, weight_decay=weight_decay)  # Our optimizer
    

    f_scores = []
    losses = []
    for epoch in range(1,epochs+1):
        print('.', end='')
    
        loss_value = train(radius, alphas, optimizer, K)
        #print("Epoch {}, Loss: {}".format(epoch, loss_value))
        p,r,f = test(radius, alphas, K, Y)
        f_scores.append(f)
        losses.append(loss_value)
    
    #plt.plot(range(1,epochs+1), f_scores, label="lambda = " + str(weight_decay))
    plt.plot(range(1,epochs+1), losses, label="lambda = " + str(weight_decay))


# Generate toy data that has two distinct classes and a huge gap between them
X, Y = make_blobs(n_samples=[5,95], n_features=5, centers=None, random_state=0, cluster_std=0.4)  # X - features, Y - labels

for weight_decay in torch.logspace(start=-2, end=0, steps = 5):
    run_experiment(X,Y, weight_decay=weight_decay)

plt.legend()
plt.show()