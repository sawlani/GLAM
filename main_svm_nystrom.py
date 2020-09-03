import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
#import matplotlib.pyplot as plt

from tqdm import tqdm


from util import load_synthetic_data, load_chem_data, load_synthetic_data_contaminated
from mmd_util import compute_mmd_gram_matrix
from models.graphcnn_svdd import GraphCNN_SVDD
from models.svm import SVM

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def train(args, model, svm, device, train_graphs, model_optimizer, svm_optimizer, epoch, k = 40):
    model.train()
    
    N = len(train_graphs)
    
    a = list(range(N))
    np.random.shuffle(a)
    Z_index = a[:k]
    Z = [train_graphs[i] for i in Z_index]

    Z_embeddings = model(Z, 1)
    
    all_vertex_embeddings = torch.cat(Z_embeddings, axis=0).detach()
    gamma = 1/torch.median(torch.cdist(all_vertex_embeddings, all_vertex_embeddings)**2)
    K_Z = compute_mmd_gram_matrix(Z_embeddings, gamma=gamma)
    eigenvalues, U_Z = torch.symeig(K_Z, eigenvectors=True)
    T = torch.matmul(U_Z,torch.diag(eigenvalues**-0.5))

    
    dists_batch_list = []

    loss_accum = 0

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]
        batch_graph = [train_graphs[idx] for idx in selected_idx]
        
        R_embeddings = model(batch_graph,1)
        K_RZ = compute_mmd_gram_matrix(R_embeddings, Z_embeddings, gamma=gamma)
        F = torch.matmul(K_RZ, T)
        
        output = svm(F).flatten()

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)
        labels[labels == 0] = -1  # Replace zeros with -1
        
        losses = torch.clamp(1 - output * labels, min=0) # hinge loss (unregularized)
        loss = torch.mean(losses)

        #backprop
        svm_optimizer.zero_grad()
        model_optimizer.zero_grad()
        
        loss.backward()
        
        svm_optimizer.step()
        model_optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        pbar.set_description('epoch: %d' % (epoch))
        
    average_loss = loss_accum/total_iters
    
    return average_loss


###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, svm, graphs, Z_embeddings, T, gamma, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        embeddings = model([graphs[j] for j in sampled_idx], 1)
        K_RZ = compute_mmd_gram_matrix(embeddings, Z_embeddings, gamma=gamma)
        F = torch.matmul(K_RZ, T)
        output.append(svm(F).flatten())
    return torch.cat(output, 0)

def test(args, model, svm, device, test_graphs, k):
    model.eval()

    N = len(test_graphs)
    
    a = list(range(N))
    np.random.shuffle(a)
    Z_index = a[:k]
    Z = [test_graphs[i] for i in Z_index]

    Z_embeddings = model(Z, 1)
    
    all_vertex_embeddings = torch.cat(Z_embeddings, axis=0).detach()
    gamma = 1/torch.median(torch.cdist(all_vertex_embeddings, all_vertex_embeddings)**2)
    K_Z = compute_mmd_gram_matrix(Z_embeddings, gamma=gamma)
    eigenvalues, U_Z = torch.symeig(K_Z, eigenvectors=True)
    T = torch.matmul(U_Z,torch.diag(eigenvalues**-0.5))

    
    output = pass_data_iteratively(model, svm, test_graphs, Z_embeddings, T, gamma)
    
    pred = torch.sign(output)
    
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    labels[labels == 0] = -1
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))
    
    return acc_test

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch GIN+MMD for whole-graph classification')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='input batch size for training (default: 10)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--dont_learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--dataset', type = str, default = "mixhop", choices=["mixhop", "chem"],
                                        help='dataset used')
    args = parser.parse_args()

    #set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if args.dataset == "mixhop":
        graphs, num_classes = load_synthetic_data(number_of_graphs=400, h_inlier=0, h_outlier=1)
        
    elif args.dataset == "contaminated":
        graphs, num_classes = load_synthetic_data_contaminated(100)
    else:
        graphs, num_classes = load_chem_data()

    ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    #train_graphs, test_graphs = separate_data(graphs,args.seed,args.fold_idx, 2)
    train_graphs, test_graphs = graphs[:100] + graphs[300:], graphs[100:300]

    k=int(0.4*len(train_graphs))
    no_of_node_features = train_graphs[0].node_features.shape[1]

    model = GraphCNN_SVDD(args.num_layers, no_of_node_features, args.hidden_dim, num_classes, (not args.dont_learn_eps), args.neighbor_pooling_type, device).to(device)
    svm = SVM(k, bias=True)

    model_optimizer = optim.SGD(model.parameters(), lr=args.lr)
    svm_optimizer = optim.SGD(svm.parameters(), lr=args.lr)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


    for epoch in range(1, args.epochs + 1):
        
        avg_loss = train(args, model, svm, device, train_graphs, model_optimizer, svm_optimizer, epoch, k)
        print("Training loss: %f" % (avg_loss))
        
        #scheduler.step()

        acc_train = test(args, model, svm, device, train_graphs, k)
        acc_test = test(args, model, svm, device, test_graphs, k)
        print("accuracy train: %f test: %f" % (acc_train, acc_test))
  
if __name__ == '__main__':
    main()
