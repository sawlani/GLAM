import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
#import matplotlib.pyplot as plt

from tqdm import tqdm

from util import load_synthetic_data, load_chem_data, load_synthetic_data_contaminated
from models.graphcnn import GraphCNN

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

criterion = nn.CrossEntropyLoss()

def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    N = len(train_graphs)
    
    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(N)[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        #compute loss
        loss = criterion(output, labels)

        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()
        

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        #report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum/total_iters
    
    return average_loss

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def test(args, model, device, test_graphs):
    model.eval()

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))
    
    return acc_test

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
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
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
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
    args = parser.parse_args()

    #set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if args.dataset == "mixhop":
        graphs, num_classes = load_synthetic_data(number_of_graphs=args.no_of_graphs, h_inlier=0.4, h_outlier=0.6)
        
    elif args.dataset == "contaminated":
        graphs, num_classes = load_synthetic_data_contaminated(number_of_graphs=args.no_of_graphs)
    else:
        graphs, num_classes = load_chem_data()

    ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    #train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)
    graphs = np.random.permutation(graphs)

    train_graphs, test_graphs = graphs[:args.no_of_graphs//2], graphs[args.no_of_graphs//2:]

    no_of_node_features = train_graphs[0].node_features.shape[1]

    model = GraphCNN(args.num_layers, no_of_node_features, args.hidden_dim, num_classes, args.final_dropout, (not args.dont_learn_eps), args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


    for epoch in range(1, args.epochs + 1):
        
        avg_loss = train(args, model, device, train_graphs, optimizer, epoch)
        print("Training loss: %f" % (avg_loss))
        
        #scheduler.step()

        acc_train = test(args, model, device, train_graphs)
        acc_test = test(args, model, device, test_graphs)
        print("accuracy train: %f test: %f" % (acc_train, acc_test))

if __name__ == '__main__':
    main()
