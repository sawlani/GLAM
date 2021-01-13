import torch, os
import numpy as np
import random
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader, DenseDataLoader
from torch_geometric.utils import degree
from pathlib import Path
import math
import pickle

from utils import load_synthetic_data

DATA_PATH = 'datasets'
if not os.path.isdir(DATA_PATH):
    os.mkdir(DATA_PATH)

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

class DownsamplingFilter(object):
    def __init__(self, min_nodes, max_nodes, down_class, down_rate, num_classes, reverse=True, coin=np.random.default_rng()):
        super(DownsamplingFilter, self).__init__()
        # if not reverse, downsampling mentioned class, and mentioned class as anomaly class
        # if reverse, downsampling unmentioned class, and mentioned class as normal class
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.down_class = down_class
        self.down_rate = down_rate
        self.reverse = reverse
        self.coin = coin
        

    def __call__(self, data):
        # step 1: filter the graph node size
        keep = (data.num_nodes <= self.max_nodes) and (data.num_nodes >= self.min_nodes)
        # for graph classification, down_rate is 1
        # downsampling only for anomaly detection, not for classification
        if self.down_rate == 1:
            return keep
        if keep:
            # step 2: downsampling class
            mentioned_class = (data.y.item() == self.down_class)
            
            anomalous_class = not mentioned_class if self.reverse else mentioned_class
            data.y.fill_(int(anomalous_class)) # anomalous class as positive

            if anomalous_class:
                if self.coin.random() > self.down_rate:
                    keep = False
        return keep

def load_data(data_name, down_class=0, down_rate=1, dense=False, classify=False, ignore_edge_weight=True, one_class_train=False, seed=1213):
    np.random.seed(seed)
    newcoin = np.random.default_rng(seed)
    torch.manual_seed(seed)
    
    if os.path.exists(data_name + "_" + str(seed) + ".pkl"):
        with open(data_name + "_" + str(seed) + ".pkl", 'rb') as f:
            dataset_raw = pickle.load(f)
    elif data_name == 'mixhop':
        NUM = 500
        dataset_raw = load_synthetic_data(num_train=NUM, num_test_inlier=NUM, num_test_outlier=int(NUM*down_rate), seed=seed)
        with open(data_name + "_" + str(seed) + ".pkl", 'wb') as f:
            pickle.dump(dataset_raw, f)
    elif data_name == 'mixhop_hard':
        NUM = 500
        dataset_raw = load_synthetic_data(num_train=NUM, num_test_inlier=NUM, num_test_outlier=int(NUM*down_rate), seed=seed, type1="mixhop-contaminated", type2="mixhop", h_inlier=0.5, h_outlier=0.5)
        with open(data_name + "_" + str(seed) + ".pkl", 'wb') as f:
            pickle.dump(dataset_raw, f)
    elif data_name in ['MNIST', 'CIFAR10']:
        dataset_raw = GNNBenchmarkDataset(root=DATA_PATH, name=data_name)
    elif 'ogbg' in data_name:
        # dataset_raw = PygGraphPropPredDataset(root=DATA_PATH, name=data_name)
        pass
        # problem: OGB needs an encoder for transform the discrete feature
        # if all embeddding are learned to be the same, then it's highly possible
        # that one-class classification will not be successful
    else:
        use_node_attr = True if data_name == 'FRANKENSTEIN' else False
        dataset_raw = TUDataset(root=DATA_PATH, name=data_name, use_node_attr=use_node_attr)
    
    
    
    # downsampling 
    # 1. Get min and max node and filter them
    num_nodes_graphs = [data.num_nodes for data in dataset_raw]
    min_nodes, max_nodes = min(num_nodes_graphs), max(num_nodes_graphs)
    if max_nodes >= 10000:
        max_nodes = 10000
    #print("min nodes, max nodes:", min_nodes, max_nodes)
    #print("#labels: ", dataset_raw.num_classes)
    
    
    # 2. Calculate down_rate for multi_class dataset: h
    from copy import deepcopy
    orig_dataset = deepcopy(dataset_raw)

    inliers = [data for data in dataset_raw if data.y == down_class]
    Ni = len(inliers)
    No = len(dataset_raw) - Ni

    down_rate = 0.5*down_rate*Ni/No
    #print("down rate:", down_rate)

    filter = DownsamplingFilter(min_nodes, max_nodes, down_class, down_rate, dataset_raw.num_classes, reverse=True, coin=newcoin)
    indices = [i for i, data in enumerate(dataset_raw) if filter(data)]
    #dataset = dataset_raw[torch.tensor(indices)].shuffle()
    dataset = dataset_raw[torch.tensor(indices)]
    
    # report the proportion info of the dataset
    n = (len(dataset) + 9) // 10
    #print("Downsampled distribution of classes in dataset:")
    #labels = np.array([data.y.item() for data in dataset])
    #label_dist = ['%4d'% (labels==c).sum() for c in range(dataset.num_classes)]
    #print("Dataset: %s, Number of graphs: %d, Class distribution %s, Num of Features %d"%(
    #        data_name, len(dataset), label_dist, dataset.num_features))

    # preprocessing: do not use original edge features or weights
    if ignore_edge_weight:
        dataset.data.edge_attr = None

    # add transforms which will be conducted when draw each elements from the dataset
    if dataset.data.x is None:
        # print('!!!')
        max_degree = 0
        degs = []
        for data in dataset_raw: # ATTENTION: use dataset_raw instead of downsampled version!
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
            # dataset.num_features = max_degree
        else:
            # dataset['num_features'] = 1
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)
    if dense:
        if dataset.transform is None:
            dataset.transform = T.ToDense(max_nodes)
        else:
            dataset.transform = T.Compose([dataset.transform, T.ToDense(max_nodes)])

    # now let's transform in memory before feed into dataloader to save runtime
    # because there is no random transformations
    dataset_list = [data for data in dataset]
    

    # here is for anomaly detection
    if not classify and not one_class_train: 
        m = 9 
        # if data_name == 'ENZYMES': m =7
        train_dataset = dataset_list[:m*n] # 90% train
        val_dataset = dataset_list[m*n:]
        test_dataset = dataset_list

    elif not classify and one_class_train:
        indices = [i for i, data in enumerate(dataset_list) if data.y.item()==0 and newcoin.random()<0.5]
        #train_dataset = train_dataset[torch.tensor(indices)] # only keep normal class left
        train_dataset = [dataset_list[idx] for idx in indices] # only keep normal class left
        val_dataset = []
        test_dataset = [dataset_list[idx] for idx in range(len(dataset_list)) if idx not in indices]
    else:
        # 10% test, 10% vali, 80% train
        test_dataset = dataset_list[:n] 
        val_dataset = dataset_list[n:2 * n]
        train_dataset = dataset_list[2 * n:]

    return train_dataset, val_dataset, test_dataset, dataset, dataset_raw, orig_dataset


def create_loaders(data_name, batch_size=32, down_class=0, down_rate=1, dense=False, classify=False, one_class_train=True, data_seed=1213, landmark_seed=0):

    train_dataset, val_dataset, test_dataset, dataset, dataset_raw, orig_dataset = load_data(data_name, 
                                                        down_class=down_class, 
                                                        down_rate=down_rate, 
                                                        dense=dense, 
                                                        classify=classify, 
                                                        one_class_train=one_class_train, seed=data_seed)
    
    k = int(5*math.log2(len(train_dataset)))
    random.seed(landmark_seed)
    landmark_set = random.sample(train_dataset, k)

    #sizes = [g.x.shape[0] for g in landmark_set]
    #print(max(sizes))

    '''
    print("Original distribution of classes in dataset:")
    labels = np.array([data.y.item() for data in orig_dataset])
    label_dist = ['%d'% (labels==c).sum() for c in range(orig_dataset.num_classes)]
    print("Dataset: %s, Number of graphs: %d, Class distribution %s, Num of (one-hot encoded) features %d"%(
            data_name, len(orig_dataset), label_dist, orig_dataset.num_features))
    
    print("Initial distribution of classes in dataset:")
    labels = np.array([data.y.item() for data in dataset_raw])
    label_dist = ['%d'% (labels==c).sum() for c in range(dataset_raw.num_classes)]
    print("Dataset: %s, Number of graphs: %d, Class distribution %s, Num of (one-hot encoded) features %d"%(
            data_name, len(dataset_raw), label_dist, dataset_raw.num_features))
    
    print("Downsampled distribution of classes in dataset:")
    labels = np.array([data.y.item() for data in dataset])
    label_dist = ['%d'% (labels==c).sum() for c in [0,1]]
    print("Dataset: %s, Number of graphs: %d, Class distribution %s, Num of (one-hot encoded) features %d"%(
            data_name, len(dataset), label_dist, dataset.num_features))
    '''
    print("After downsampling and test-train splitting, distribution of classes:")
    labels = np.array([data.y.item() for data in train_dataset])
    label_dist = ['%d'% (labels==c).sum() for c in [0,1]]
    print("TRAIN: Number of graphs: %d, Class distribution %s"%(
            len(train_dataset), label_dist))
    
    #print("After downsampling and test-train splitting, distribution of classes in TEST dataset:")
    labels = np.array([data.y.item() for data in test_dataset])
    label_dist = ['%d'% (labels==c).sum() for c in [0,1]]
    print("TEST: Number of graphs: %d, Class distribution %s"%(
            len(test_dataset), label_dist))
    

    Loader = DenseDataLoader if dense else DataLoader
    num_workers = 0
    train_loader = Loader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = Loader(test_dataset, batch_size=batch_size, shuffle=False,  pin_memory=True, num_workers=num_workers)
    landmark_loader = Loader(landmark_set, batch_size=batch_size, shuffle=False,  pin_memory=True, num_workers=num_workers)

    
    return train_loader, test_loader, landmark_loader, dataset_raw[0].num_features