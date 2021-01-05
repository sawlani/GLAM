import torch
from torch.nn.utils.rnn import pad_sequence


def compute_gamma(embeddings, device=torch.device("cpu")):
    all_vertex_embeddings = torch.cat(embeddings, axis=0).detach().to(device)
    all_vertex_distances = torch.cdist(all_vertex_embeddings, all_vertex_embeddings)**2
    median_of_distances = torch.median(all_vertex_distances)
    if median_of_distances <= 1e-4:
        median_of_distances = torch.tensor(1e-4).to(device)
    
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
    X_ones_padded = pad_sequence(X_ones, batch_first=True).to(device)
    del X_ones
    Y_ones = [torch.ones(emb.shape[0]) for emb in Y_embeddings]
    Y_ones_padded = pad_sequence(Y_ones, batch_first=True).to(device)
    del Y_ones
    mask = X_ones_padded[:,None,:,None]*Y_ones_padded[None,:,None,:]
    del X_ones_padded
    del Y_ones_padded

    XY = torch.matmul(X_padded[:,None,:,:], torch.transpose(Y_padded[None,:,:,:], -1, -2))
    
    if type=="SMM":
        X_sq = torch.squeeze(torch.matmul(X_padded[:,:,None,:], X_padded[:,:,:,None]))
        del X_padded
        Y_sq = torch.squeeze(torch.matmul(Y_padded[:,:,None,:], Y_padded[:,:,:,None]))
        del Y_padded
        K_XY = torch.exp(-gamma * (-2 * XY + X_sq[:,None,:,None] + Y_sq[None,:,None,:]))
        del XY
        del X_sq
        del Y_sq

        masked_means = torch.true_divide(torch.sum(K_XY*mask,(2,3)), torch.sum(mask,(2,3)))
    else:
        raise ValueError("This type is not supported (yet)")
    
    return masked_means