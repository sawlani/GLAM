import torch
from torch.nn.utils.rnn import pad_sequence

def compute_gamma(embeddings, device=torch.device("cpu")):
    all_vertex_embeddings = torch.cat(embeddings, axis=0).detach().to(device)
    all_vertex_distances = torch.cdist(all_vertex_embeddings, all_vertex_embeddings)**2
    median_of_distances = torch.median(all_vertex_distances)
    if median_of_distances <= 1e-4:
        median_of_distances = 1e-4
    
    gamma = 1/median_of_distances

    return gamma

def compute_mmd_gram_matrix(X_embeddings, Y_embeddings=None, gamma=None, type="SMM", device=torch.device("cpu")):
    
    if not Y_embeddings:
        Y_embeddings = X_embeddings

    if gamma == None:
        gamma = compute_gamma(Y_embeddings)
    if gamma==0:
        raise ValueError("Gamma value appears to be 0")
    
    # pad with 0s and convert to 3d tensor. 
    X_padded = pad_sequence(X_embeddings, batch_first=True).to(device)
    Y_padded = pad_sequence(Y_embeddings, batch_first=True).to(device)

    # calculate mask to be able to exclude padded 0s later while computing mean
    X_ones = [torch.ones(emb.shape[0]) for emb in X_embeddings]
    Y_ones = [torch.ones(emb.shape[0]) for emb in Y_embeddings]
    X_ones_padded = pad_sequence(X_ones, batch_first=True).to(device)
    Y_ones_padded = pad_sequence(Y_ones, batch_first=True).to(device)
    mask = X_ones_padded[:,None,:,None]*Y_ones_padded[None,:,None,:]

    XY = torch.matmul(X_padded[:,None,:,:], torch.transpose(Y_padded[None,:,:,:], -1, -2))

    if type=="SMM":
        X_sq = torch.squeeze(torch.matmul(X_padded[:,:,None,:], X_padded[:,:,:,None]))
        Y_sq = torch.squeeze(torch.matmul(Y_padded[:,:,None,:], Y_padded[:,:,:,None]))
        
        K_XY = torch.exp(-gamma * (-2 * XY + X_sq[:,None,:,None] + Y_sq[None,:,None,:]))

        masked_means = torch.sum(K_XY*mask,(2,3))/torch.sum(mask,(2,3))
    else:
        raise ValueError("This type is not supported (yet)")
    
    return masked_means




''' Old code below - bot being used'''

def rbf_mmd(X, Y, gamma=None, type="SMM"):

    if gamma == None:
        gamme = compute_gamma([X,Y])
    
    XY = torch.matmul(X, torch.transpose(Y, 0, 1))
    
    if type=="MMD":
        XX = torch.matmul(X, torch.transpose(X, 0, 1))
        YY = torch.matmul(Y, torch.transpose(Y, 0, 1))
        X_sqnorms = torch.diagonal(XX)
        Y_sqnorms = torch.diagonal(YY)

        K_XY = torch.exp(-gamma * (-2 * XY + torch.unsqueeze(X_sqnorms,1) + torch.unsqueeze(Y_sqnorms,0)))
        K_XX = torch.exp(-gamma * (-2 * XX + torch.unsqueeze(X_sqnorms,1) + torch.unsqueeze(X_sqnorms,0)))
        K_YY = torch.exp(-gamma * (-2 * YY + torch.unsqueeze(Y_sqnorms,1) + torch.unsqueeze(Y_sqnorms,0)))

        output = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()

    elif type=="SMM":
        X_sqnorms = torch.squeeze(torch.matmul(torch.unsqueeze(X,1),torch.unsqueeze(X,2)))
        Y_sqnorms = torch.squeeze(torch.matmul(torch.unsqueeze(Y,1),torch.unsqueeze(Y,2)))
        
        K_XY = torch.exp(-gamma * (-2 * XY + torch.unsqueeze(X_sqnorms,1) + torch.unsqueeze(Y_sqnorms,0)))
        
        output = K_XY.mean()

    return output
    
def rbf_mmd_simple(X, Y, gamma, type="SMM"):
    if gamma == None:
        gamme = compute_gamma([X,Y])
    return torch.mean(torch.exp(-1*gamma* torch.cdist(X,Y)**2))

def compute_mmd_gram_matrix_slow(X_embeddings, Y_embeddings=None, gamma=None, type="SMM"):
    
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
                MMD_values[i][j] = rbf_mmd(X_embeddings[i], Y_embeddings[j], gamma=gamma, type=type)
    
    else:
        n = len(X_embeddings)
        MMD_values = torch.zeros(n,n)
        
        for i in range(n):
            for j in range(i,n):
                MMD_values[i][j] = rbf_mmd(X_embeddings[i], X_embeddings[j], gamma=gamma, type=type)
                MMD_values[j][i] = MMD_values[i][j]
    
    return MMD_values
