import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, average_precision_score


from util import load_synthetic_data, load_chem_data, separate_data
from mmd_util import compute_mmd_gram_matrix
from models.graphcnn_svdd import GraphCNN_SVDD

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

R = torch.tensor(0)
def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)

'''
def train(args, model, device, train_graphs, optimizer, epoch, k, variance_multiplier, layer="all"):
    model.eval()
    
    Z = np.random.permutation(train_graphs)[:k]
    
    Z_embeddings = model(Z, layer)

    
    all_vertex_embeddings = torch.cat(Z_embeddings, axis=0).detach()
    gamma = 1/torch.median(torch.cdist(all_vertex_embeddings, all_vertex_embeddings)**2)
    
    K_Z = compute_mmd_gram_matrix(Z_embeddings, gamma=gamma)
    
    eigenvalues, U_Z = torch.symeig(K_Z, eigenvectors=True)
    T = torch.matmul(U_Z,torch.diag(eigenvalues**-0.5))

    F_Z = torch.matmul(U_Z,torch.diag(eigenvalues**0.5))

    approx_center = torch.median(F_Z, dim=0).values

    #F_list = []
    
    dists_batch_list = []
    #total_loss = 0
    no_of_batches = 0

    mean_sum = 0
    mean_squared_sum = 0



    for start in range(0, len(train_graphs), args.batch_size):
        print(".", end='')
        no_of_batches += 1
        batch_graph = train_graphs[start:start+args.batch_size]

        R_embeddings = model(batch_graph,layer)
        K_RZ = compute_mmd_gram_matrix(R_embeddings, Z_embeddings, gamma=gamma)
        F = torch.matmul(K_RZ, T)
        
        mean_sum += torch.mean(K_RZ)
        mean_squared_sum += torch.mean(K_RZ**2)

        dists_batch = torch.sum((F - approx_center)**2, dim=1)
        dists_batch_list.append(dists_batch)
        

    variance = (mean_squared_sum/no_of_batches) - (mean_sum/no_of_batches)**2
    dists = torch.cat(dists_batch_list, axis=0)
    
    loss1 = torch.mean(dists)
    loss2 =  - variance
    
    print(loss1, loss2)

    loss = loss1 + loss2*variance_multiplier

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        
    
    loss = loss.detach().cpu().numpy()
    
    return loss
'''


def train(args, model, device, train_graphs, optimizer, epoch, k, variance_multiplier, layer="all"):
    model.eval()
    
    loss_accum = 0
    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    
    for pos in pbar:
        Z = np.random.permutation(train_graphs)[:k]
        Z_embeddings = model(Z, layer)

        all_vertex_embeddings = torch.cat(Z_embeddings, axis=0).detach()
        gamma = 1/torch.median(torch.cdist(all_vertex_embeddings, all_vertex_embeddings)**2)
        
        K_Z = compute_mmd_gram_matrix(Z_embeddings, gamma=gamma)        
        eigenvalues, U_Z = torch.symeig(K_Z, eigenvectors=True)
        T = torch.matmul(U_Z,torch.diag(eigenvalues**-0.5))

        F_Z = torch.matmul(U_Z,torch.diag(eigenvalues**0.5))
        approx_center = torch.median(F_Z, dim=0).values

        

        batch_graph = np.random.permutation(train_graphs)[:args.batch_size]
        
        R_embeddings = model(batch_graph,layer)
        K_RZ = compute_mmd_gram_matrix(R_embeddings, Z_embeddings, gamma=gamma)
        F = torch.matmul(K_RZ, T)
        
        dists = torch.sum((F - approx_center)**2, dim=1)
        loss = torch.mean(dists)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        pbar.set_description('epoch: %d' % (epoch))
        

    average_loss = loss_accum/total_iters
    
    return average_loss

def test(args, model, device, test_graphs, k, layer="all"):
    model.eval()
    
    Z = np.random.permutation(test_graphs)[:k]
    
    Z_embeddings = model(Z, layer)

    
    all_vertex_embeddings = torch.cat(Z_embeddings, axis=0).detach()
    gamma = 1/torch.median(torch.cdist(all_vertex_embeddings, all_vertex_embeddings)**2)
    
    K_Z = compute_mmd_gram_matrix(Z_embeddings, gamma=gamma)
    
    eigenvalues, U_Z = torch.symeig(K_Z, eigenvectors=True)
    T = torch.matmul(U_Z,torch.diag(eigenvalues**-0.5))

    F_Z = torch.matmul(U_Z,torch.diag(eigenvalues**0.5))

    approx_center = torch.median(F_Z, dim=0).values

    dists_batch_list = []
    
    for start in range(0, len(test_graphs), args.batch_size):
        #print(".", end='')
        
        batch_graph = test_graphs[start:start+args.batch_size]

        R_embeddings = model(batch_graph,layer)
        K_RZ = compute_mmd_gram_matrix(R_embeddings, Z_embeddings, gamma=gamma)
        F = torch.matmul(K_RZ, T)
        
        dists_batch = torch.sum((F - approx_center)**2, dim=1)
        dists_batch_list.append(dists_batch)
        

    dists = torch.cat(dists_batch_list, axis=0)
    #print(dists)
    dists = dists.detach().cpu().numpy()
    
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    
    score = average_precision_score(labels, dists)
    return score

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    #parser.add_argument('--dataset', type=str, default="MUTAG",
    #                    help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",
                        help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type = str, default = "",
                                        help='output file')
    parser.add_argument('--dataset', type = str, default = "mixhop", choices=["mixhop", "chem", "contaminated"],
                                        help='dataset used')
    parser.add_argument('--no_of_graphs', type = int, default = 100,
                                        help='no of graphs generated')
    parser.add_argument('--layer', type = str, default = "all",
                                        help='which hidden layer used as embedding')
    parser.add_argument('--h_inlier', type=float, default=0.3,
                        help='inlier homophily (default: 0.3)')
    parser.add_argument('--h_outlier', type=float, default=0.7,
                        help='inlier homophily (default: 0.7)')
    args = parser.parse_args()

    #set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if args.layer != "all":
        args.layer = int(args.layer)

    if args.dataset == "mixhop":
        graphs, num_classes = load_synthetic_data(number_of_graphs=args.no_of_graphs, h_inlier=args.h_inlier, h_outlier=args.h_outlier, outlier_ratio=0.05)
        
    elif args.dataset == "contaminated":
        graphs, num_classes = load_synthetic_data_contaminated(number_of_graphs=args.no_of_graphs, outlier_ratio=0.05)
    else:
        graphs, num_classes = load_chem_data()

    train_graphs, test_graphs = graphs, graphs

    k_frac = 0.4
    k = int(k_frac*len(train_graphs))
    no_of_node_features = train_graphs[0].node_features.shape[1]

    for variance_multiplier in [0,1,2,4,8]:
    
        model = GraphCNN_SVDD(args.num_layers, no_of_node_features, args.hidden_dim, num_classes, args.learn_eps, args.neighbor_pooling_type, device).to(device)
    

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        aps = []


        #PRE-TRAINING TEST
        score = test(args, model, device, graphs, k, layer=args.layer)
        print("Pre-Training AP Score: %f" % score)
        aps.append(score)


        for epoch in range(1, args.epochs + 1):
            
            avg_loss = train(args, model, device, graphs, optimizer, epoch, k, variance_multiplier, layer=args.layer)
            print("Training loss: %f" % (avg_loss))
            
            #scheduler.step()

            score = test(args, model, device, graphs, k, layer=args.layer)
            
            print("Avg Precision Score: %f" % score)
            aps.append(score)

        plt.plot(list(range(0, args.epochs + 1)), aps, label="varmul="+str(variance_multiplier))

    plt.grid()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()