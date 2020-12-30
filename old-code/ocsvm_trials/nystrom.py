import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.svm import OneClassSVM
import numpy as np
import numpy.linalg as LA
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
    K = torch.exp(K)  # exponentiate K in-place
    return K




# Generate toy data that has two distinct classes and a huge gap between them
X, Y = make_blobs(n_samples=[5,95], centers=None, random_state=0, cluster_std=0.4)  # X - features, Y - labels

X = torch.Tensor(X)

k = 10

a = list(range(len(X)))
np.random.shuffle(a)
Z_index = a[:k]

Z = X[Z_index]

K_Z = rbf_kernel(Z, gamma = 1)


eigenvalues, U_Z = LA.eigh(K_Z)

SIG_Z = np.diag(eigenvalues)

SQRTSIG_Z = np.diag(eigenvalues**0.5)

