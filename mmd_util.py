import torch
import numpy as np
from sklearn import metrics


def rbf_mmd(X, Y, sigma="median", biased=True):

    if sigma == 'median':
        median_samples = 1000
        from sklearn.metrics.pairwise import euclidean_distances
        sub = lambda feats, n: feats[np.random.choice(
            feats.shape[0], min(feats.shape[0], n), replace=False)]
        #Z = np.r_[sub(X, median_samples // 2), sub(Y, median_samples // 2)]
        Z = torch.cat((sub(X, median_samples // 2), sub(Y, median_samples // 2)),0)
        D2 = torch.cdist(Z,Z,2)**2
        #D2 = euclidean_distances(Z, squared=True)
        upper = D2[np.triu_indices_from(D2, k=1)]
        kernel_width = torch.median(upper)
        gamma = 1/kernel_width
        # sigma = median / sqrt(2); works better, sometimes at least
        del Z, D2, upper
    else:
        gamma = 1 / (2 * sigma**2)
    
    XX = torch.mm(X, torch.transpose(X, 0, 1))
    XY = torch.mm(X, torch.transpose(Y, 0, 1))
    YY = torch.mm(Y, torch.transpose(Y, 0, 1))
    X_sqnorms = torch.diagonal(XX)
    Y_sqnorms = torch.diagonal(YY)

    K_XY = torch.exp(-gamma * (-2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
    K_XX = torch.exp(-gamma * (-2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]))
    K_YY = torch.exp(-gamma * (-2 * YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))

    if biased:
        mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    else:
        m = K_XX.shape[0]
        n = K_YY.shape[0]

        mmd = ((K_XX.sum() - m) / (m * (m - 1))
              + (K_YY.sum() - n) / (n * (n - 1))
              - 2 * K_XY.mean())
    return mmd


def compute_mmd_gram_matrix(embeddings, width=None):
    
    if not width:
        width = "median"

    n = len(embeddings)
    MMD_values = torch.zeros((n,n))
    
    for i in range(n):
        for j in range(i,n):

            MMD_values[i][j] = rbf_mmd(embeddings[i], embeddings[j], width)
            MMD_values[j][i] = MMD_values[i][j]
    
    return MMD_values

def mmd_score(graphs, MMD_scores):
    l = len(graphs)
    labels = torch.zeros_like(MMD_scores)
    #print(l, MMD_scores.shape)

    for i in range(l):
        for j in range(l):
            if graphs[i].label != graphs[j].label:
                labels[i,j] = 1
    
    MMD_scores = torch.flatten(MMD_scores)
    labels = torch.flatten(labels)
    
    return metrics.roc_auc_score(labels, MMD_scores)

