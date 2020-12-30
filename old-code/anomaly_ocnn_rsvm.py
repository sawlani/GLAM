import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import matplotlib.pyplot as plt

from util import load_synthetic_data, load_chem_data, load_synthetic_data_contaminated
from mmd_util import compute_mmd_gram_matrix, compute_gamma

from models.graphcnn_svdd import GraphCNN_SVDD
from models.svm import SVM

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def train_bigstep(args, model, svm, device, train_graphs, model_optimizer, svm_optimizer, epoch, Z):
    model.train()
    
    Z_embeddings = model(Z, args.layer)
    Z_embeddings = [emb.detach() for emb in Z_embeddings]
    gamma = compute_gamma(Z_embeddings)
    
    loss_accum = 0
    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    model_optimizer.zero_grad()
    
    for pos in pbar:
        
        batch_graph = np.random.permutation(train_graphs)[:args.batch_size]
        
        R_embeddings = model(batch_graph,args.layer)
        K_RZ = compute_mmd_gram_matrix(R_embeddings, Z_embeddings, gamma=gamma, type="SMM")

        dists = svm(K_RZ).flatten()

        if epoch > args.warm_up_n_epochs:
            args.margin = np.sqrt(np.quantile(dists.clone().data.cpu().numpy(), args.nu))
        
        
        scores = torch.clamp(args.margin - dists, min=0)
        loss = (1/args.nu)*torch.mean(scores)

        svm_optimizer.zero_grad()

        loss.backward(retain_graph=True)

        svm_optimizer.step()
        
        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        pbar.set_description('epoch: %d' % (epoch))
        

    model_optimizer.step()
    
    average_loss = loss_accum/total_iters
    
    return average_loss


def test(args, model, svm, device, test_graphs, Z):
    model.eval()
    
    Z_embeddings = model(Z, args.layer)
    gamma = compute_gamma(Z_embeddings)
    
    dists_batch_list = []
    for start in range(0, len(test_graphs), args.batch_size):
        #print(".", end='')
        
        batch_graph = test_graphs[start:start+args.batch_size]

        R_embeddings = model(batch_graph,args.layer)
        K_RZ = compute_mmd_gram_matrix(R_embeddings, Z_embeddings, gamma=gamma)
        
        dists_batch = svm(K_RZ).flatten()
        dists_batch_list.append(dists_batch)
        
    dists = torch.cat(dists_batch_list, axis=0)
    dists = dists.detach().cpu().numpy()
    print(dists)
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    
    score = average_precision_score(labels, -dists) # Negative because in this case, outliers have SMALLER distance values
    return score, dists
    
def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
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
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--weight_decay', type=float, default=1,
                        help='weight_decay constant (lambda), default=1.')
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
    parser.add_argument('--no_of_graphs', type = int, default = 200,
                                        help='no of graphs generated')
    parser.add_argument('--layer', type = str, default = "all",
                                        help='which hidden layer used as embedding')
    parser.add_argument('--h_inlier', type=float, default=0.3,
                        help='inlier homophily (default: 0.3)')
    parser.add_argument('--h_outlier', type=float, default=0.7,
                        help='outlier homophily (default: 0.7)')
    parser.add_argument('--nu', type=float, default=0.05,
                        help='expected fraction of outliers (default: 0.05)')
    parser.add_argument('--k_frac', type=float, default=0.2,
                        help='fraction of landmark points for RSVM (default: 0.2)')
    parser.add_argument('--margin', type=float, default=0,
                        help='hypersphere radius (default: 0)')
    parser.add_argument('--warm_up_n_epochs', type=float, default=10,
                        help='epochs before radius is updated (default: 10)')
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

    k = int(args.k_frac*args.no_of_graphs)

    Z = np.random.permutation(graphs)[:k]  #landmark set
    no_of_node_features = graphs[0].node_features.shape[1]

    margins = [4]
    
    APS = []
    OUTLIER_RATIOS = []

    TRAINING_LOSSES = []
    NON_REG_LOSSES = []
    SVM_REG_LOSSES = []
    MODEL_REG_LOSSES = []


    for margin in margins:
        args.margin = margin
        print("Margin=%f" % margin)

        model = GraphCNN_SVDD(args.num_layers, no_of_node_features, args.hidden_dim, num_classes, (not args.dont_learn_eps), args.neighbor_pooling_type, device).to(device)
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
        score, dists = test(args, model, svm, device, graphs, Z)
        print("Pre-Training AP Score: %f" % score)
        
        no_epochs = args.epochs
        for epoch in range(1, no_epochs + 1):
        
            train = train_bigstep
            
            avg_loss = train(args, model, svm, device, graphs, model_optimizer, svm_optimizer, epoch, Z)
            
            model_reg_loss = 0
            for param in model.parameters():
                model_reg_loss += 0.5 * args.weight_decay * torch.sum(param ** 2)
            
            svm_reg_loss = 0
            for param in svm.parameters():
                svm_reg_loss += 0.5 * args.weight_decay * torch.sum(param ** 2)
            
            print("Training loss: %f + %f + %f + %f = %f" % (-args.margin, avg_loss, svm_reg_loss, model_reg_loss, -args.margin+avg_loss+svm_reg_loss+model_reg_loss))
            
            #scheduler.step()

            score, dists = test(args, model, svm, device, graphs, Z)
            
            print("Avg Precision Score: %f" % score)

            outlier_ratio = sum(dists < args.margin)/len(dists)

            print("Outlier ratio: %f" % outlier_ratio)

            aps.append(score)
            outlier_ratios.append(outlier_ratio)
            
            training_losses.append(avg_loss)
            non_reg_losses.append(avg_loss - args.margin)
            svm_reg_losses.append(svm_reg_loss)
            model_reg_losses.append(model_reg_loss)
            
        fig, axs = plt.subplots(3,2, sharex=True)
        fig.suptitle("OCNN for fixed margin=%f" % args.margin)
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
        
        fig.savefig("margin"+str(args.margin)+".png", dpi=1500)

        APS.append(score)
        OUTLIER_RATIOS.append(outlier_ratio)
        
        TRAINING_LOSSES.append(avg_loss)
        NON_REG_LOSSES.append(avg_loss - args.margin)
        SVM_REG_LOSSES.append(svm_reg_loss)
        MODEL_REG_LOSSES.append(model_reg_loss)

        
    fig, axs = plt.subplots(3,2, sharex=True)
    fig.suptitle("OCNN for fixed margin=%f" % args.margin)
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