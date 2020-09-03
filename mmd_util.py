import torch
import numpy as np
from sklearn import metrics


def rbf_mmd_old(X, Y, gamma="median"):

    if gamma == 'median':
        median_samples = 100
        from sklearn.metrics.pairwise import euclidean_distances
        sub = lambda feats, n: feats[np.random.choice(
            feats.shape[0], min(feats.shape[0], n), replace=False)]
        
        Z = torch.cat((sub(X, median_samples // 2), sub(Y, median_samples // 2)),0)
        D2 = torch.cdist(Z,Z,2)**2
        upper = D2[np.triu_indices_from(D2, k=1)]
        kernel_width = torch.median(upper)
        gamma = 1/kernel_width
        del Z, D2, upper
    
    XX = torch.mm(X, torch.transpose(X, 0, 1))
    XY = torch.mm(X, torch.transpose(Y, 0, 1))
    YY = torch.mm(Y, torch.transpose(Y, 0, 1))
    X_sqnorms = torch.diagonal(XX)
    Y_sqnorms = torch.diagonal(YY)

    K_XY = torch.exp(-gamma * (-2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
    #K_XY = torch.exp(-gamma * (torch.cdist(X,Y)**2))
    #K_XY = XY**2
    
    mmd = K_XY.mean()

    return mmd
    
def rbf_mmd(X, Y, gamma="median"):

    if gamma == 'median':
        median_samples = 100
        from sklearn.metrics.pairwise import euclidean_distances
        sub = lambda feats, n: feats[np.random.choice(
            feats.shape[0], min(feats.shape[0], n), replace=False)]
        Z = torch.cat((sub(X, median_samples // 2), sub(Y, median_samples // 2)),0)
        D2 = torch.cdist(Z,Z,2)**2
        upper = D2[np.triu_indices_from(D2, k=1)]
        kernel_width = torch.median(upper)
        gamma = 1/kernel_width
        del Z, D2, upper
    
    return torch.mean(torch.exp(-gamma* torch.cdist(X,Y)**2))

def compute_mmd_gram_matrix(X_embeddings, Y_embeddings=None, gamma='median'):
    
    if Y_embeddings:
        n1 = len(X_embeddings)
        n2 = len(Y_embeddings)
        MMD_values = torch.zeros(n1,n2)
        
        for i in range(n1):
            for j in range(n2):
                MMD_values[i][j] = rbf_mmd_old(X_embeddings[i], Y_embeddings[j], gamma)
    
    else:
        n = len(X_embeddings)
        MMD_values = torch.zeros(n,n)
        
        for i in range(n):
            for j in range(i,n):
                MMD_values[i][j] = rbf_mmd_old(X_embeddings[i], X_embeddings[j], gamma)
                MMD_values[j][i] = MMD_values[i][j]
    
        if not (torch.symeig(MMD_values).eigenvalues > 0).all():
            print("Matrix not Positive Definite")

    return MMD_values
