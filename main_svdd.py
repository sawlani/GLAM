import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, average_precision_score


from util import load_synthetic_data, load_chem_data, separate_data
from mmd_util import compute_mmd_gram_matrix, mmd_score
from models.graphcnn_svdd import GraphCNN_SVDD

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def train(args, model, device, train_graphs, optimizer, epoch, lmbd=1):
    
    model.train()
    
    batch_graph = train_graphs
    output= model(batch_graph)
    print(output)
    #radius = model.radius
    #print(radius)
    
    #weight = model.svm.weight.squeeze()
    
    #labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    #labels[labels == 0] = -1  # Replace zeros with -1

    #labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)    

    #output = torch.flatten(output)
    #losses = torch.clamp(1 - output * labels, min=0) # hinge loss (unregularized)

    loss = torch.mean(output) # + (lmbd/2)*(radius**2)

    #loss += torch.dot(weight,weight)/ 2.0

    
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
    
        optimizer.step()
        

    loss = loss.detach().cpu().numpy()
    
    return loss

def get_hidden_layer(model, graphs, layer):
    model.eval()

    embeddings = []
    for graph in graphs:

        embedding = model.get_hidden_rep([graph], layer)[0]
        embeddings.append(embedding)
        
    return embeddings

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
#def pass_data_iteratively(model, graphs, minibatch_size = 64):
#    model.eval()
#    output = []
#    idx = np.arange(len(graphs))
#    for i in range(0, len(graphs), minibatch_size):
#        sampled_idx = idx[i:i+minibatch_size]
#        if len(sampled_idx) == 0:
#            continue
#        output.append(model([graphs[j] for j in sampled_idx]).detach())
#    return torch.cat(output, 0)

def test(args, model, device, test_graphs):
    model.eval()

    '''
    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))
    '''

    output = model(test_graphs)
    scores = output.detach().numpy()
    #preds = (output == 0)
    
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    

    #p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary")
    score = average_precision_score(labels, scores)
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

    ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    #train_graphs, test_graphs = separate_data(graphs,args.seed,args.fold_idx, 2)
    train_graphs, test_graphs = graphs, graphs

    model = GraphCNN_SVDD(len(train_graphs), args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)
    

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


    for epoch in range(1, args.epochs + 1):
        
        avg_loss = train(args, model, device, train_graphs, optimizer, epoch)
        print("Training loss: %f" % (avg_loss))
        
        scheduler.step()

        score = test(args, model, device, test_graphs)
        #print("Precision: %f, Recall: %f, F-1 score: %f" % (p, r,f))
        print("Avg Precision Score: %f" % score)


    #for layer in range(args.num_layers):
    #    print("Hidden Layer: %d" % (layer+1))
    #    embeddings = get_hidden_layer(model, graphs, layer)
    #    for width in [0.032, 0.1, 0.32, 1, 3.2, 10, 32, 100, 320, 1000]:
    #        MMD_values = compute_mmd_gram_matrix(embeddings, width).detach()
    #        print(mmd_score(graphs, MMD_values))

if __name__ == '__main__':
    main()
