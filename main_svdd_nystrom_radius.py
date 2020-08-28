import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, average_precision_score


from util import load_synthetic_data, load_chem_data, separate_data
from mmd_util import compute_mmd_gram_matrix
from models.graphcnn_svdd_nystrom import GraphCNN_SVDD

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)

def train(args, model, device, train_graphs, optimizer, epoch, k = 20, batch_size = 10, radius, nu):
    model.eval()
    
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

    F_Z = torch.matmul(U_Z,torch.diag(eigenvalues**0.5))

    approx_center = torch.mean(F_Z, dim=0)

    #F_list = []
    
    dists_batch_list = []

    for start in range(0, N, batch_size):
        print(".", end='')
        batch_graph = train_graphs[start:start+batch_size]

        R_embeddings = model(batch_graph,1)
        K_RZ = compute_mmd_gram_matrix(R_embeddings, Z_embeddings, gamma=gamma)
        F = torch.matmul(K_RZ, T)
        
        #F_list.append(F)

        dists_batch = torch.sum((F - approx_center)**2, dim=1)
        dists_batch_list.append(dists_batch)

    #F_full = torch.cat(F_list, axis=0)

    #center = torch.mean(F_full, dim=0)

    #dists = torch.sum((F_full - center)**2, dim=1)
    dists = torch.cat(dists_batch_list, axis=0)
    
    print(dists)
    
    scores = torch.clamp(dists - (radius**2), min=0)

    #weight = model.svm.weight.squeeze()
    #loss = 20*torch.mean(scores) + (R**2)

    loss = torch.mean(scores)
    #loss += torch.dot(weight,weight)/ 2.0

    
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
    
        optimizer.step()
        
    #if epoch >= 3:
    #    R.data = torch.tensor(get_radius(dists, 0.05))
    
    loss = loss.detach().cpu().numpy()
    dists = dists.detach().cpu().numpy()
    return loss, dists

def get_hidden_layer(model, graphs, layer):
    model.eval()

    embeddings = []
    for graph in graphs:

        embedding = model.get_hidden_rep([graph], layer)[0]
        embeddings.append(embedding)
        
    return embeddings


def test(args, dists, device, test_graphs):
    #model.eval()

    '''
    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))
    '''

    #output = model(test_graphs)
    #scores = output.detach().numpy()
    #preds = (scores > 0)
    
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    

    #p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary")
    score = average_precision_score(labels, dists)
    #return p,r,f
    return score

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    #parser.add_argument('--dataset', type=str, default="MUTAG",
    #                    help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='input batch size for training (default: 1)')
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
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",
                        help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type = str, default = "",
                                        help='output file')
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
        graphs, num_classes = load_synthetic_data(100,0.05)
    else:
        graphs, num_classes = load_chem_data()

    train_graphs, test_graphs = graphs, graphs

    R = torch.tensor(0)

    for k in range(20,30,10):
    
        model = GraphCNN_SVDD(len(train_graphs), args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)
    

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        aps = []
        for epoch in range(1, args.epochs + 1):
            
            avg_loss, dists = train(args, model, device, train_graphs, optimizer, epoch, k, batch_size=10, radius=R, nu=0.05)
            print("Training loss: %f" % (avg_loss))
            
            scheduler.step()

            score = test(args, dists, device, test_graphs)
            #print("Precision: %f, Recall: %f, F-1 score: %f" % (p, r,f))
            print("Avg Precision Score: %f" % score)
            aps.append(score)

        plt.plot(list(range(1, args.epochs + 1)), aps, label="k="+str(k))

    plt.grid()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
