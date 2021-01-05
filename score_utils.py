import numpy as np
from sklearn.metrics import calinski_harabasz_score

def mod_CH(X, nu=0.05):
    X = np.sort(X)
    
    outlier_mean = np.mean(X[int((1-nu)*len(X)):])
    for i in range(int((1-nu)*len(X)), len(X)):
        X[i] = outlier_mean
    
    labels = np.array([0]*int((1-nu)*len(X)) + [1]*(len(X) - int((1-nu)*len(X))))
    X = X.reshape(-1, 1)
    score = calinski_harabasz_score(X, labels)
    return score