import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import matplotlib.pyplot as plt

from util import load_synthetic_data, load_chem_data, separate_data
from mmd_util import compute_mmd_gram_matrix

from models.graphcnn_svdd import GraphCNN_SVDD
from models.svm import SVM

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#def get_margin(dist: torch.Tensor, nu: float):
    #"""Optimally solve for margin R via the (1-nu)-quantile of distances."""
    #return np.quantile(dist.clone().data.cpu().numpy(), nu)


def train_smallstep(args, model, svm, device, train_graphs, model_optimizer, svm_optimizer, epoch, k, margin=0, nu=0.05, layer="all"):
    model.train()
    
    loss_accum = 0
    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    
    for pos in pbar:
        Z = np.random.permutation(train_graphs)[:k]
        Z_embeddings = model(Z, layer)

        all_vertex_embeddings = torch.cat(Z_embeddings, axis=0).detach()
        all_vertex_distances = torch.cdist(all_vertex_embeddings, all_vertex_embeddings)**2
        median_of_distances = torch.median(all_vertex_distances)
        if median_of_distances == 0:
            median_of_distances = torch.median(all_vertex_distances[all_vertex_distances>0])
        
        gamma = 1/median_of_distances
    
        K_Z = compute_mmd_gram_matrix(Z_embeddings, gamma=gamma)        
        eigenvalues, U_Z = torch.symeig(K_Z, eigenvectors=True)
        eigenvalues = torch.clamp(eigenvalues, min=1e-5)
        T = torch.matmul(U_Z,torch.diag(eigenvalues**-0.5))
    
        batch_graph = np.random.permutation(train_graphs)[:args.batch_size]
        
        R_embeddings = model(batch_graph,layer)
        K_RZ = compute_mmd_gram_matrix(R_embeddings, Z_embeddings, gamma=gamma)
        F = torch.matmul(K_RZ, T)
        
        dists = svm(F).flatten()
        scores = torch.clamp(margin - dists, min=0)
        loss = (1/nu)*torch.mean(scores)

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


def train_bigstep(args, model, svm, device, train_graphs, model_optimizer, svm_optimizer, epoch, k, margin=0, nu=0.05, layer="all"):
    model.train()
    

    Z = np.random.permutation(train_graphs)[:k]
    Z_embeddings = model(Z, layer)

    all_vertex_embeddings = torch.cat(Z_embeddings, axis=0).detach()
    all_vertex_distances = torch.cdist(all_vertex_embeddings, all_vertex_embeddings)**2
    median_of_distances = torch.median(all_vertex_distances)
    if median_of_distances <= 1e-4:
        median_of_distances = torch.min(all_vertex_distances[all_vertex_distances>1e-4])
    
    gamma = 1/median_of_distances
    
    K_Z = compute_mmd_gram_matrix(Z_embeddings, gamma=gamma)
    eigenvalues, U_Z = torch.symeig(K_Z, eigenvectors=True)
    eigenvalues = torch.clamp(eigenvalues, min=1e-5)

    T = torch.matmul(U_Z,torch.diag(eigenvalues**-0.5))

    loss_accum = 0
    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    model_optimizer.zero_grad()
    
    for pos in pbar:
        

        batch_graph = np.random.permutation(train_graphs)[:args.batch_size]
        
        R_embeddings = model(batch_graph,layer)
        K_RZ = compute_mmd_gram_matrix(R_embeddings, Z_embeddings, gamma=gamma)
        F = torch.matmul(K_RZ, T)
        
        dists = svm(F).flatten()
        scores = torch.clamp(margin - dists, min=0)
        loss = (1/nu)*torch.mean(scores)

        svm_optimizer.zero_grad()

        loss.backward(retain_graph=True)

        svm_optimizer.step()
        
        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        pbar.set_description('epoch: %d' % (epoch))
        

    model_optimizer.step()
    
    average_loss = loss_accum/total_iters
    
    return average_loss


def test(args, model, svm, device, test_graphs, k, layer="all"):
    model.eval()
    
    Z = np.random.permutation(test_graphs)[:k]
    Z_embeddings = model(Z, layer)

    all_vertex_embeddings = torch.cat(Z_embeddings, axis=0).detach()
    all_vertex_distances = torch.cdist(all_vertex_embeddings, all_vertex_embeddings)**2
    median_of_distances = torch.median(all_vertex_distances)
    if median_of_distances <= 1e-5:
        median_of_distances = torch.min(all_vertex_distances[all_vertex_distances>1e-5])
    
    gamma = 1/median_of_distances
    
    
    K_Z = compute_mmd_gram_matrix(Z_embeddings, gamma=gamma)
    eigenvalues, U_Z = torch.symeig(K_Z, eigenvectors=True)
    eigenvalues = torch.clamp(eigenvalues, min=1e-5)

    T = torch.matmul(U_Z,torch.diag(eigenvalues**-0.5))

    dists_batch_list = []
    
    for start in range(0, len(test_graphs), args.batch_size):
        #print(".", end='')
        
        batch_graph = test_graphs[start:start+args.batch_size]

        R_embeddings = model(batch_graph,layer)
        K_RZ = compute_mmd_gram_matrix(R_embeddings, Z_embeddings, gamma=gamma)
        F = torch.matmul(K_RZ, T)
        
        dists_batch = svm(F).flatten()
        dists_batch_list.append(dists_batch)
        

    dists = torch.cat(dists_batch_list, axis=0)
    #print(dists)
    dists = dists.detach().cpu().numpy()
    if np.isnan(dists).any():
        print("what's going on")
    
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    
    
    score = average_precision_score(labels, -dists) # Negative because in this case, outliers have SMALLER distance values
    return score, dists
    
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
    parser.add_argument('--iters_per_epoch', type=int, default=20,
                        help='number of iterations per each epoch (default: 20)')
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
    parser.add_argument('--weight_decay', type=float, default=1,
                        help='weight_decay constant (lambda), default=1.')
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
    parser.add_argument('--no_of_graphs', type = int, default = 200,
                                        help='no of graphs generated')
    parser.add_argument('--layer', type = str, default = "all",
                                        help='which hidden layer used as embedding')
    parser.add_argument('--h_inlier', type=float, default=0.3,
                        help='inlier homophily (default: 0.3)')
    parser.add_argument('--h_outlier', type=float, default=0.7,
                        help='outlier homophily (default: 0.7)')
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

    #train_graphs, test_graphs = graphs, graphs

    k_frac = 0.2
    k = int(k_frac*len(graphs))
    no_of_node_features = graphs[0].node_features.shape[1]

    margins = [4,8,16,32,64,128,256]
    
    APS = []
    OUTLIER_RATIOS = []

    TRAINING_LOSSES = []
    NON_REG_LOSSES = []
    SVM_REG_LOSSES = []
    MODEL_REG_LOSSES = []


    for margin in margins:
        print("Margin=%f" % margin)

        model = GraphCNN_SVDD(args.num_layers, no_of_node_features, args.hidden_dim, num_classes, args.learn_eps, args.neighbor_pooling_type, device).to(device)
        model_optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        svm = SVM(k)
        svm_optimizer = optim.SGD(svm.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        aps = []
        outlier_ratios = []

        training_losses = []
        non_reg_losses = []
        svm_reg_losses = []
        model_reg_losses = []

        #PRE-TRAINING TEST
        score, dists = test(args, model, svm, device, graphs, k, layer=args.layer)
        print("Pre-Training AP Score: %f" % score)
        #aps.append(score)

        outlier_ratio = 1.0
        no_epochs = args.epochs
        for epoch in range(1, no_epochs + 1):
        #epoch=0
        #while outlier_ratio > 0.05:
            #epoch +=1

            train = train_bigstep
            #train = train_smallstep
            
            avg_loss = train(args, model, svm, device, graphs, model_optimizer, svm_optimizer, epoch, k, margin=margin, nu = 0.05, layer=args.layer)
            
            model_reg_loss = 0
            for param in model.parameters():
                model_reg_loss += 0.5 * args.weight_decay * torch.sum(param ** 2)
            
            svm_reg_loss = 0
            for param in svm.parameters():
                svm_reg_loss += 0.5 * args.weight_decay * torch.sum(param ** 2)
            
            print("Training loss: %f + %f + %f + %f = %f" % (-margin, avg_loss, svm_reg_loss, model_reg_loss, -margin+avg_loss+svm_reg_loss+model_reg_loss))
            
            #scheduler.step()

            score, dists = test(args, model, svm, device, graphs, k, layer=args.layer)
            
            print("Avg Precision Score: %f" % score)

            outlier_ratio = sum(dists < margin)/len(dists)

            print("Outlier ratio: %f" % outlier_ratio)

            aps.append(score)
            outlier_ratios.append(outlier_ratio)
            
            training_losses.append(avg_loss)
            non_reg_losses.append(avg_loss - margin)
            svm_reg_losses.append(svm_reg_loss)
            model_reg_losses.append(model_reg_loss)
            
        fig, axs = plt.subplots(3,2, sharex=True)
        fig.suptitle("OCNN for fixed margin=%f" % margin)
        fig.tight_layout(pad=2, h_pad=0.5, w_pad=2)

        axs[0,0].set(ylabel='Average Precision')
        axs[0,0].plot(list(range(1, no_epochs + 1)), aps)
        axs[0,0].grid()

        axs[0,1].set(ylabel='Outlier Ratio')
        axs[0,1].plot(list(range(1, no_epochs + 1)), outlier_ratios)
        axs[0,1].grid()

        axs[1,0].set(ylabel='Training Loss')
        axs[1,0].plot(list(range(1, no_epochs + 1)), training_losses)
        axs[1,0].grid()

        axs[1,1].set(ylabel='SVM Reg Loss')
        axs[1,1].plot(list(range(1, no_epochs + 1)), svm_reg_losses)
        axs[1,1].grid()

        axs[2,0].set(xlabel='Epochs', ylabel='Non-reg Loss')
        axs[2,0].plot(list(range(1, no_epochs + 1)), non_reg_losses)
        axs[2,0].grid()

        axs[2,1].set(xlabel='Epochs', ylabel='GIN Reg Loss')
        axs[2,1].plot(list(range(1, no_epochs + 1)), model_reg_losses)
        axs[2,1].grid()
        
        fig.savefig("margin"+str(margin)+".png", dpi=1500)

        APS.append(score)
        OUTLIER_RATIOS.append(outlier_ratio)
        
        TRAINING_LOSSES.append(avg_loss)
        NON_REG_LOSSES.append(avg_loss - margin)
        SVM_REG_LOSSES.append(svm_reg_loss)
        MODEL_REG_LOSSES.append(model_reg_loss)

        
    fig, axs = plt.subplots(3,2, sharex=True)
    fig.suptitle("OCNN for fixed margin=%f" % margin)
    fig.tight_layout(pad=2, h_pad=0.5, w_pad=2)

    axs[0,0].set(ylabel='Average Precision')
    axs[0,0].plot(margins, APS)
    axs[0,0].grid()

    axs[0,1].set(ylabel='Outlier Ratio')
    axs[0,1].plot(margins, OUTLIER_RATIOS)
    axs[0,1].grid()

    axs[1,0].set(ylabel='Training Loss')
    axs[1,0].plot(margins, TRAINING_LOSSES)
    axs[1,0].grid()

    axs[1,1].set(ylabel='SVM Reg Loss')
    axs[1,1].plot(margins, SVM_REG_LOSSES)
    axs[1,1].grid()

    axs[2,0].set(xlabel='Margins', ylabel='Non-reg Loss')
    axs[2,0].plot(margins, NON_REG_LOSSES)
    axs[2,0].grid()

    axs[2,1].set(xlabel='Margins', ylabel='GIN Reg Loss')
    axs[2,1].plot(margins, MODEL_REG_LOSSES)
    axs[2,1].grid()
    
    fig.savefig("OCNN_loss_vs_margin.png", dpi=1500)
    plt.show()

if __name__ == '__main__':
    main()