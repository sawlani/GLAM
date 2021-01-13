#main.py

import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import math
import pickle
import argparse

from dataloader import create_loaders
from GIN import GIN
from trainers import MMDTrainer, MeanTrainer
from utils import mod_CH

from matplotlib import rcParams
rcParams.update({'figure.autolayout': False})

from types import SimpleNamespace

def run_experiment(
    data = "saved", data_seed=1213, down_cls=0, down_rate=0.05,
    alpha=1.0, beta=0.0, epochs=150, model_seed=0, landmark_seed=100, num_layers=1,
    device=0, aggregation="MMD", nystrom="LLSVM", bias=False, hidden_dim=64, lr=0.2, weight_decay=1e-5, batch = 64
    ):

    filename = ""
    filename += data
    filename += str(data_seed)
    filename += "_seed_m"
    filename += str(model_seed)
    filename += "l"
    filename += str(landmark_seed)
    filename += "_"
    filename += "LR"
    filename += str(lr)
    filename += "_"
    filename += "WD"
    filename += "%.1E" %(weight_decay)
    filename += "_"
    filename += str(num_layers)
    filename += "lyr"
    #filename += "_a"
    #filename += str(alpha)
    #filename += "_b"
    #filename += str(beta)
    if bias:
        filename += "_bias"
    else:
        filename += "_nobias"
    filename += ".png"

    
    device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")
    
    # load data
    train_loader, test_loader, landmark_loader, num_features = create_loaders(data_name=data, 
                            batch_size=batch, 
                            down_class=down_cls, 
                            down_rate=down_rate, 
                            dense=False,
                            data_seed=data_seed,
                            landmark_seed=landmark_seed)

    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)

    model = GIN(nfeat = num_features, nhid=hidden_dim, nlayer=num_layers, bias=bias)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    if aggregation=="MMD":

        trainer = MMDTrainer(
            model=model,
            optimizer=optimizer,
            landmark_loader=landmark_loader,
            alpha=alpha,
            beta=beta,
            device=device,
            nystrom=nystrom
            )
    
    elif aggregation=="Mean":
        trainer = MeanTrainer(
            model=model,
            optimizer=optimizer,
            alpha=alpha,
            beta=beta,
            device=device
            )
    
    total_losses = []
    epochinfo = []
    converged_epoch = -1

    for epoch in range(epochs+1):

        print("Epoch %3d" % (epoch), end="\t")
        loss, svdd_loss, reg_loss = trainer.train(train_loader=train_loader)
        print("SVDD loss: %f" % (svdd_loss), end="\t")
        print("Total loss: %f" % (loss), end="\t")
        ap, roc_auc, dists, labels = trainer.test(test_loader=test_loader)
        print("AP: %f" % ap, end="\t")
        print("ROC-AUC: %f" % roc_auc)

        
        TEMP = SimpleNamespace()
        TEMP.epoch_no = epoch
        
        TEMP.dists = dists
        TEMP.labels = labels
        TEMP.ap = ap
        TEMP.roc_auc = roc_auc
        TEMP.svdd_loss = svdd_loss
        TEMP.reg_loss = reg_loss
        TEMP.total_loss = loss

        TEMP.ch2 = mod_CH(dists)

        epochinfo.append(TEMP)
        total_losses.append(loss)

        if epoch > 10 and converged_epoch < 0:
            if max(total_losses[-5:]) <= 1.05* min(total_losses[1:]):
                converged_epoch = epoch
                print("converged at epoch: %d" % (epoch))
            

    # SAVE INFO TO RETURN ############################
    if converged_epoch == -1:
        converged_epoch = len(epochinfo)-1

    ch2s_copy = [e.ch2 for e in epochinfo]
    best_idx = np.argmax(np.array(ch2s_copy))
    print("       Max CH2, at epoch %d, AP: %.2f, ROC-AUC: %.2f" % (best_idx, epochinfo[best_idx].ap, epochinfo[best_idx].roc_auc))
    print("At convergence, at epoch %d, AP: %.2f, ROC-AUC: %.2f" % (converged_epoch, epochinfo[converged_epoch].ap, epochinfo[converged_epoch].roc_auc))
    print("    At the end, at epoch %d, AP: %.2f, ROC-AUC: %.2f" % (args.epochs, epochinfo[-1].ap, epochinfo[-1].roc_auc))

    important_epoch_info = {}
    important_epoch_info['external'] = epochinfo[best_idx]
    important_epoch_info['converged'] = epochinfo[converged_epoch]
    important_epoch_info['last'] = epochinfo[-1]
    #################################################
    
    
    # Plotting graph for run ############################
    gap = max(1,epochs//40)
    intermittent_epochinfo = epochinfo[::gap]

    inlier_xs = []
    inlier_ys = []
    outlier_xs = []
    outlier_ys = []

    for i, e in enumerate(intermittent_epochinfo):
        for dist, label in zip(e.dists, e.labels):
            if label == 0:
                inlier_xs.append(gap*i)
                inlier_ys.append(dist)
            else:
                outlier_xs.append(gap*i)
                outlier_ys.append(dist)

    
    fig, axs = plt.subplots(5, figsize=(15,15), sharex=True)

    i = 0

    axs[i].set(ylabel='AP', ylim=((0,1)))
    axs[i].plot(list(range(epochs+1)), [e.ap for e in epochinfo])
    axs[i].grid()
    axs[i].xaxis.set_tick_params(labelbottom=True)
    axs[i].axvline(x=best_idx, ls=':', color='green', lw=2)
    axs[i].axvline(x=converged_epoch, ls=':', color='magenta', lw=2)

    i += 1
    axs[i].set(ylabel='ROC-AUC', ylim=((0,1)))
    axs[i].plot(list(range(epochs+1)), [e.roc_auc for e in epochinfo])
    axs[i].grid()
    axs[i].xaxis.set_tick_params(labelbottom=True)
    axs[i].axvline(x=best_idx, ls=':', color='green', lw=2)
    axs[i].axvline(x=converged_epoch, ls=':', color='magenta', lw=2)

    i += 1
    axs[i].set(ylabel='Dists')
    axs[i].scatter(inlier_xs, inlier_ys, s=10, color='blue', alpha=0.2)
    axs[i].scatter(outlier_xs, outlier_ys, s=20, color='red', alpha=0.2)
    axs[i].grid()
    axs[i].xaxis.set_tick_params(labelbottom=True)
    axs[i].axvline(x=best_idx, ls=':', color='green', lw=2)
    axs[i].axvline(x=converged_epoch, ls=':', color='magenta', lw=2)

    i += 1
    axs[i].set(ylabel='Total loss')
    axs[i].plot(list(range(1,epochs+1)), [e.total_loss for e in epochinfo[1:]])
    axs[i].grid()
    axs[i].xaxis.set_tick_params(labelbottom=True)
    axs[i].axvline(x=best_idx, ls=':', color='green', lw=2)
    axs[i].axvline(x=converged_epoch, ls=':', color='magenta', lw=2)

    i += 1
    axs[i].set(xlabel='Epochs', ylabel='mCH')
    axs[i].plot(list(range(epochs+1)), [e.ch2 for e in epochinfo])
    axs[i].grid()
    axs[i].xaxis.set_tick_params(labelbottom=True)
    axs[i].axvline(x=best_idx, ls=':', color='green', lw=2)
    axs[i].axvline(x=converged_epoch, ls=':', color='magenta', lw=2)

    fig.savefig(filename, dpi=128)
    
    f = plt.figure()
    f.clear()
    plt.close(f)
    #################################################
    
    return important_epoch_info


parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')

parser.add_argument('--data', default='mixhop',
                    help='dataset name (default: mixhop)')
parser.add_argument('--batch', type=int, default=64,
                    help='batch size (default: 64)')
parser.add_argument('--data_seed', type=int, default=1213,
                    help='seed to split the inlier set into train and test (default: 1213)')
parser.add_argument('--down_cls', type=int, default=0,
                    help='inlier class (default: 0)')
parser.add_argument('--down_rate', type=float, default=0.05,
                    help='outlier/inlier fraction (default: 0.05)')


parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--epochs', type=int, default=150,
                    help='number of epochs to train (default: 150)')


parser.add_argument('--hidden_dim', type=int, default=64,
                    help='number of hidden units (default: 64)')
parser.add_argument('--bias', action="store_true",
                                    help='Whether to use bias terms in the GNN.')

parser.add_argument('--aggregation', type=str, default="MMD", choices=["MMD", "Mean"],
                    help='Type of graph level aggregation (default: MMD)')
parser.add_argument('--nystrom', type=str, default="LLSVM", choices=["LLSVM", "RSVM"], 
                    help='Type of Kernel Mapping for Nystrom (default: LLSVM)')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='regularizer: 1 - speed of adaptivity')
parser.add_argument('--beta', type=float, default=0.0,
                    help='regularizer loss multiplier (ratio)')


parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight_decay constant lambda (default: 1e-4)')
parser.add_argument('--landmark_seed', type=int, default=666, 
                    help='First landmark seed (default: 666)')
parser.add_argument('--landmark_seed_number', type=int, default=3, 
                    help='Number of landmark seeds (default: 3)')
parser.add_argument('--model_seed', type=int, default=0, 
                    help='First model seed (default: 0)')
parser.add_argument('--model_seed_number', type=int, default=3, 
                    help='Number of model seeds (default: 3)')

args = parser.parse_args()

D = {}

lrs = [args.lr, args.lr*2]
weight_decays = [args.weight_decay, args.weight_decay*10]
model_seeds = [args.model_seed+i for i in range(args.model_seed_number)]
landmark_seeds = [args.landmark_seed+i for i in range(args.landmark_seed_number)]
layercounts = [1,2,4]

for lr in lrs:
    for weight_decay in weight_decays:
        for model_seed in model_seeds:
            for landmark_seed in landmark_seeds:
                for layercount in layercounts:
                    print("Running experiment for LR=%f, weight decay = %.1E, landmark seed = %d, model seed = %d, number of layers = %d" % (lr, weight_decay, landmark_seed, model_seed, layercount))
                    info = run_experiment(
                        data=args.data,
                        data_seed=args.data_seed,
                        down_cls=args.down_cls,
                        down_rate=args.down_rate,
                        alpha=args.alpha,
                        beta=args.beta,
                        epochs=args.epochs,
                        model_seed=model_seed,
                        landmark_seed=landmark_seed,
                        num_layers=layercount,
                        device=args.device,
                        aggregation=args.aggregation,
                        nystrom=args.nystrom,
                        bias=args.bias,
                        hidden_dim=args.hidden_dim,
                        lr=lr,
                        weight_decay=weight_decay,
                        batch=args.batch
                        )
                    
                    D[(lr,weight_decay,landmark_seed,model_seed, layercount)] = info

with open('all_models_' + args.data + '_' + str(args.data_seed) + '.pkl', 'wb') as f:
    pickle.dump(D, f)