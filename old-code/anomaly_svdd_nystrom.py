import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torch.optim as optim

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

import argparse
from tqdm import tqdm

from util import load_synthetic_data, load_chem_data, load_synthetic_data_contaminated
from mmd_util import compute_mmd_gram_matrix, compute_gamma
from models.graphcnn_svdd import GraphCNN_SVDD

def train_bigstep(args, model, train_graphs, optimizer, epoch, Z):
    model.train()
    
    Z_embeddings = model(Z, args.layer)
    Z_embeddings = [emb.detach() for emb in Z_embeddings]
    
    gamma = compute_gamma(Z_embeddings, device=args.device)
    
    K_Z = compute_mmd_gram_matrix(Z_embeddings, gamma=gamma, device=args.device).to(args.device)
    eigenvalues, U_Z = torch.symeig(K_Z, eigenvectors=True)
    T = torch.matmul(U_Z,torch.diag(eigenvalues**-0.5))

    F_Z = torch.matmul(U_Z,torch.diag(eigenvalues**0.5))
    approx_center = torch.median(F_Z, dim=0).values
    #approx_center = torch.mean(F_Z, dim=0)

    loss_accum = 0
    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    optimizer.zero_grad()
    
    for pos in pbar:

        batch_graph = np.random.permutation(train_graphs)[:args.batch_size]
        
        R_embeddings = model(batch_graph,args.layer)
        K_RZ = compute_mmd_gram_matrix(R_embeddings, Z_embeddings, gamma=gamma, device=args.device).to(args.device)
        
        F = torch.matmul(K_RZ, T)
        
        dists = torch.sum((F - approx_center)**2, dim=1).cpu()
        
        # Update hypersphere radius R on mini-batch distances
        if epoch > args.warm_up_n_epochs:
            args.radius = np.sqrt(np.quantile(dists.clone().data.cpu().numpy(), 1 - args.nu))
        
        scores = torch.clamp(dists - (args.radius**2), min=0)

        
        ''''
        reconstruction_loss_accum = 0
        for embedding,graph in zip(R_embeddings, batch_graph):
            reconstructed = torch.zeros(embedding.shape[0], embedding.shape[0])
            for i,v1 in enumerate(embedding):
                for j,v2 in enumerate(embedding):
                    reconstructed[i,j] = decoder(v1,v2)
            
            #reconstructed = torch.sigmoid(torch.matmul(embedding, torch.transpose(embedding, 0, 1)))
            actual = nx.adjacency_matrix(graph.g)
            actual = actual.tocoo()
            actual = torch.sparse.LongTensor(torch.LongTensor([actual.row.tolist(), actual.col.tolist()]),
                              torch.LongTensor(actual.data.astype(np.int32)))
            reconstruction_loss_accum += torch.norm(reconstructed-actual, p="fro")
        
        reconstruction_loss = reconstruction_loss_accum/args.batch_size
        '''
        #batch_variance = torch.mean(F.std(dim=0)**2)
        #batch_variance_loss = torch.clamp(args.d - batch_variance, min=0)
        
        #kernel_variance = torch.var(K_RZ)

        svdd_loss = (1/args.nu)*torch.mean(scores)

        loss = svdd_loss
        #loss = svdd_loss + reconstruction_loss
        #print(svdd_loss, kernel_variance, torch.mean(K_RZ))
        
        loss.backward(retain_graph=True)
        
        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        pbar.set_description('epoch: %d' % (epoch))
        

    optimizer.step()
    
    average_loss = loss_accum/total_iters
    
    return average_loss

def test(args, model, test_graphs, Z):
    model.eval()
    
    Z_embeddings = model(Z, args.layer)
    gamma = compute_gamma(Z_embeddings, args.device)
    
    K_Z = compute_mmd_gram_matrix(Z_embeddings, gamma=gamma, device=args.device).to(args.device)
    
    eigenvalues, U_Z = torch.symeig(K_Z, eigenvectors=True)
    T = torch.matmul(U_Z,torch.diag(eigenvalues**-0.5))

    F_Z = torch.matmul(U_Z,torch.diag(eigenvalues**0.5))

    approx_center = torch.median(F_Z, dim=0).values
    #approx_center = torch.mean(F_Z, dim=0)

    dists_batch_list = []
    
    for start in range(0, len(test_graphs), args.batch_size):
        #print(".", end='')
        
        batch_graph = test_graphs[start:start+args.batch_size]

        R_embeddings = model(batch_graph,args.layer)
        K_RZ = compute_mmd_gram_matrix(R_embeddings, Z_embeddings, gamma=gamma, device=args.device).to(args.device)
        F = torch.matmul(K_RZ, T)
        
        dists_batch = torch.sum((F - approx_center)**2, dim=1)
        dists_batch_list.append(dists_batch)
        

    dists = torch.cat(dists_batch_list, axis=0).to(args.device)
    dists = dists.detach()
    #print(dists)
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(args.device)
    
    score = average_precision_score(labels, dists)
    return score, dists

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch whole-graph anomaly detection')
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
    parser.add_argument('--dont_learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",
                        help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--dataset', type = str, default = "mixhop", choices=["mixhop", "chem", "contaminated"],
                                        help='dataset used')
    parser.add_argument('--no_of_graphs', type = int, default = 200,
                                        help='no of graphs generated')
    parser.add_argument('--layer', type = str, default = "1",
                                        help='which hidden layer used as embedding')
    parser.add_argument('--h_inlier', type=float, default=0.3,
                        help='inlier homophily (default: 0.3)')
    parser.add_argument('--h_outlier', type=float, default=0.7,
                        help='outlier homophily (default: 0.7)')
    parser.add_argument('--nu', type=float, default=0.05,
                        help='expected fraction of outliers (default: 0.05)')
    parser.add_argument('--k_frac', type=float, default=0.2,
                        help='fraction of landmark points for RSVM (default: 0.2)')
    parser.add_argument('--radius', type=float, default=0,
                        help='hypersphere radius (default: 0)')
    parser.add_argument('--warm_up_n_epochs', type=float, default=10,
                        help='epochs before radius is updated (default: 10)')
    args = parser.parse_args()

    #set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)    
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
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

    train_graphs, test_graphs = graphs[:190], graphs

    k = int(args.k_frac*args.no_of_graphs)
    Z = np.random.permutation(train_graphs)[:k]
    
    no_of_node_features = graphs[0].node_features.shape[1]

    radii = [0, 0.02, 0.04, 0.06, 0.08, 1.0]

    APS = []
    OUTLIER_RATIOS = []

    TRAINING_LOSSES = []
    NON_REG_LOSSES = []
    MODEL_REG_LOSSES = []


    for radius in radii:
        print("Radius=%f" % radius)
        args.radius = radius
        model = GraphCNN_SVDD(args.num_layers, no_of_node_features, args.hidden_dim, num_classes, (not args.dont_learn_eps), args.neighbor_pooling_type, args.device).to(args.device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        #if args.layer == "all":
        #    decoder = NTN(args.hidden_dim*args.num_layers)
        #else:
        #    decoder = NTN(args.hidden_dim)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        aps = []
        outlier_ratios = []

        training_losses = []
        non_reg_losses = []
        model_reg_losses = []

        #PRE-TRAINING TEST
        score, dists = test(args, model, test_graphs, Z)
        print("Pre-Training AP Score: %f" % score)
        
        no_epochs = args.epochs
        for epoch in range(1, no_epochs + 1):
        
            train = train_bigstep
            
            avg_loss = train(args, model, train_graphs, optimizer, epoch, Z)
            
            model_reg_loss = 0
            for param in model.parameters():
                model_reg_loss += 0.5 * args.weight_decay * torch.sum(param ** 2)
            
            print("Training loss: %f + %f + %f = %f" % (args.radius**2, avg_loss, model_reg_loss, args.radius**2 +avg_loss+model_reg_loss))
            
            #scheduler.step()

            score, dists = test(args, model, test_graphs, Z)
            
            print("Avg Precision Score: %f" % score)

            outlier_ratio = sum(dists > args.radius**2)/len(dists)

            print("Outlier ratio: %f" % outlier_ratio)

            aps.append(score)
            outlier_ratios.append(outlier_ratio)
            
            training_losses.append(avg_loss)
            non_reg_losses.append(avg_loss + args.radius**2)
            model_reg_losses.append(model_reg_loss)
            
        fig, axs = plt.subplots(2,2, sharex=True)
        fig.suptitle("SVDD with Nystrom for fixed radius=%f" % args.radius)
        fig.tight_layout(pad=2, h_pad=2, w_pad=2)

        axs[0,0].set(ylabel='Average Precision')
        axs[0,0].plot(list(range(1, no_epochs + 1)), aps)
        axs[0,0].grid()

        axs[0,1].set(ylabel='Outlier Ratio')
        axs[0,1].plot(list(range(1, no_epochs + 1)), outlier_ratios)
        axs[0,1].grid()

        axs[1,0].set(ylabel='Training Loss')
        axs[1,0].plot(list(range(1, no_epochs + 1)), training_losses)
        axs[1,0].grid()

        #axs[2,0].set(xlabel='Epochs', ylabel='Non-reg Loss')
        #axs[2,0].plot(list(range(1, no_epochs + 1)), non_reg_losses)
        #axs[2,0].grid()

        #axs[2,1].set(xlabel='Epochs', ylabel='GIN Reg Loss')
        #axs[2,1].plot(list(range(1, no_epochs + 1)), model_reg_losses)
        #axs[2,1].grid()
        
        fig.savefig("sep16_radius"+str(args.radius)+".png", dpi=1500)

        APS.append(score)
        OUTLIER_RATIOS.append(outlier_ratio)
        
        TRAINING_LOSSES.append(avg_loss)
        NON_REG_LOSSES.append(avg_loss + args.radius**2)
        MODEL_REG_LOSSES.append(model_reg_loss)

        
    fig, axs = plt.subplots(2,2, sharex=True)
    fig.suptitle("SVDD with Nystrom for fixed radius=%f" % args.radius)
    fig.tight_layout(pad=2, h_pad=2, w_pad=2)

    axs[0,0].set(ylabel='Average Precision')
    axs[0,0].plot(radii, APS)
    axs[0,0].grid()

    axs[0,1].set(ylabel='Outlier Ratio')
    axs[0,1].plot(radii, OUTLIER_RATIOS)
    axs[0,1].grid()

    axs[1,0].set(ylabel='Training Loss')
    axs[1,0].plot(radii, TRAINING_LOSSES)
    axs[1,0].grid()

    axs[1,1].set(xlabel='Radii', ylabel='Non-reg Loss')
    axs[1,1].plot(radii, NON_REG_LOSSES)
    axs[1,1].grid()

    #axs[2,1].set(xlabel='Radii', ylabel='GIN Reg Loss')
    #axs[2,1].plot(radii, MODEL_REG_LOSSES)
    #axs[2,1].grid()
    
    fig.savefig("Sep16_SVDD_loss_vs_radius.png", dpi=1500)
    plt.show()

if __name__ == '__main__':
    main()