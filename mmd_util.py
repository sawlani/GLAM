import torch
import numpy as np
from sklearn import metrics


def rbf_mmd_old(X, Y, gamma=None, type="SMM"):

    if gamma == None:
        gamme = compute_gamma([X,Y])
    
    XX = torch.mm(X, torch.transpose(X, 0, 1))
    XY = torch.mm(X, torch.transpose(Y, 0, 1))
    YY = torch.mm(Y, torch.transpose(Y, 0, 1))
    X_sqnorms = torch.diagonal(XX)
    Y_sqnorms = torch.diagonal(YY)

    
    if type=="MMD":
        K_XY = torch.exp(-gamma * (-2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
        K_XX = torch.exp(-gamma * (-2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]))
        K_YY = torch.exp(-gamma * (-2 * YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))

        output = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()

    elif type=="SMM":
        K_XY = torch.exp(-gamma * (-2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
        
        output = K_XY.mean()

    return output
    
def rbf_mmd(X, Y, gamma):
    if gamma == None:
        gamme = compute_gamma([X,Y])
    return torch.mean(torch.exp(-1*gamma* torch.cdist(X,Y)**2))

def compute_mmd_gram_matrix(X_embeddings, Y_embeddings=None, gamma=None, type="SMM"):
    
    if gamma == None:
        gamma = compute_gamma(X_embeddings)
    if gamma==0:
        print("zero gamma")

    if Y_embeddings:
        n1 = len(X_embeddings)
        n2 = len(Y_embeddings)
        MMD_values = torch.zeros(n1,n2)
        
        for i in range(n1):
            for j in range(n2):
                MMD_values[i][j] = rbf_mmd_old(X_embeddings[i], Y_embeddings[j], gamma=gamma, type=type)
    
    else:
        n = len(X_embeddings)
        MMD_values = torch.zeros(n,n)
        
        for i in range(n):
            for j in range(i,n):
                MMD_values[i][j] = rbf_mmd_old(X_embeddings[i], X_embeddings[j], gamma=gamma, type=type)
                MMD_values[j][i] = MMD_values[i][j]
    
    return MMD_values

def compute_gamma(embeddings):
    all_vertex_embeddings = torch.cat(embeddings, axis=0).detach()
    all_vertex_distances = torch.cdist(all_vertex_embeddings, all_vertex_embeddings)**2
    median_of_distances = torch.median(all_vertex_distances)
    if median_of_distances <= 1e-4:
        median_of_distances = 1e-4
    
    gamma = 1/median_of_distances

    return gamma