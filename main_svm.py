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

# Extract a hidden layer from GIN as the embedding
# Compute the SMM-kernel matrix between all graphs
# Full-batch training on the entire set of training graphs
# Each epoch has args.iters_per_epoch iterations of full-batch training
def train(args, model, svm, device, train_graphs, model_optimizer, svm_optimizer, epoch, layer="all"):
    model.train()
    
    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in pbar:

        all_embeddings = model(train_graphs,layer)
        all_vertex_embeddings = torch.cat(all_embeddings, axis=0).detach()
        gamma = 1/torch.median(torch.cdist(all_vertex_embeddings, all_vertex_embeddings)**2)

        K_full = compute_mmd_gram_matrix(all_embeddings, gamma=gamma)

        output = svm(K_full).flatten()

        labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
        labels[labels == 0] = -1  # Replace zeros with -1

        losses = torch.clamp(1 - output * labels, min=0) # hinge loss (unregularized)
        loss = torch.mean(losses)

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


def test(args, model, svm, device, test_graphs, layer="all"):
    model.eval()

    N = len(test_graphs)
    
    all_embeddings = model(test_graphs,layer)
    all_vertex_embeddings = torch.cat(all_embeddings, axis=0).detach()
    gamma = 1/torch.median(torch.cdist(all_vertex_embeddings, all_vertex_embeddings)**2)

    K_full = compute_mmd_gram_matrix(all_embeddings, gamma=gamma)

    output = svm(K_full).flatten()

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
    parser.add_argument('--dataset', type = str, default = "mixhop", choices=["mixhop", "chem", "contaminated"],
                                        help='dataset used')
    parser.add_argument('--no_of_graphs', type = int, default = 100,
                                        help='no of graphs generated')
    parser.add_argument('--layer', type = str, default = "all",
                                        help='which hidden layer used as embedding')
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
        graphs, num_classes = load_synthetic_data(number_of_graphs=args.no_of_graphs, h_inlier=0.3, h_outlier=0.7)
        
    elif args.dataset == "contaminated":
        graphs, num_classes = load_synthetic_data_contaminated(number_of_graphs=args.no_of_graphs)
    else:
        graphs, num_classes = load_chem_data()

    ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    #train_graphs, test_graphs = separate_data(graphs,args.seed,args.fold_idx, 2)
    graphs = np.random.permutation(graphs)

    train_graphs, test_graphs = graphs[:args.no_of_graphs//2], graphs[args.no_of_graphs//2:]

    no_of_node_features = train_graphs[0].node_features.shape[1]

    model = GraphCNN_SVDD(args.num_layers, no_of_node_features, args.hidden_dim, num_classes, (not args.dont_learn_eps), args.neighbor_pooling_type, device).to(device)
    svm = SVM(len(train_graphs), bias=True)

    model_optimizer = optim.SGD(model.parameters(), lr=args.lr)
    svm_optimizer = optim.SGD(svm.parameters(), lr=args.lr)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


    for epoch in range(1, args.epochs + 1):
        
        avg_loss = train(args, model, svm, device, train_graphs, model_optimizer, svm_optimizer, epoch, layer=args.layer)
        print("Training loss: %f" % (avg_loss))
        
        #scheduler.step()

        acc_train = test(args, model, svm, device, train_graphs, layer=args.layer)
        acc_test = test(args, model, svm, device, test_graphs, layer=args.layer)
        print("accuracy train: %f test: %f" % (acc_train, acc_test))
  
if __name__ == '__main__':
    main()
