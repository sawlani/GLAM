import networkx as nx
import numpy as np
import random
import torch
from sklearn.model_selection import StratifiedKFold
from graphgen import MixhopGraphGenerator
from read_chem import extract_chem_graphs
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.node_features = node_features # one-hot encoded node-tags
        self.edge_mat = None

        # edge_mat is of the form:
        # [[u1 u2 u3 ... um]
        #  [v1 v2 v3 ... vm]]


'''
def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list

'''

def random_split_counts(total_number, no_of_splits):
    split_indices = np.sort(np.random.choice(total_number,no_of_splits-1, replace = False)+1)
    split_indices = np.insert(split_indices, 0, 0)
    split_indices = np.append(split_indices, total_number)
    split_counts = np.diff(split_indices)
    return split_counts

def draw_graph(G, name):
    colors = []
    for v in G.nodes:
        color = G.nodes[v]['color']
        colors.append(color)
    
    top = [node for node in G.nodes if G.nodes[node]['color'] == colors[0]]

    nx.draw(G, pos=nx.bipartite_layout(G, top), node_color=colors, node_size=25)
    plt.draw()
    plt.savefig(name)
    plt.close()

def load_synthetic_data(number_of_graphs = 100, h_inlier=0.4, h_outlier=0.6, outlier_ratio=0.5, n_min = 50, n_max = 150, no_of_tags = 2, type1 = "mixhop", type2 = "mixhop"):
    print('generating data')
    g_list = []
    
    number_of_outliers = int(number_of_graphs*outlier_ratio)

    for i in range(number_of_graphs - number_of_outliers):
        
        n = np.random.randint(n_min, n_max)
        tag_counts = random_split_counts(n, no_of_tags)

        if type1 == "mixhop":
            g = MixhopGraphGenerator(tag_counts, heteroWeightsExponent=1.0)(n, 5, 10, h_inlier)
            tags = [g.nodes[v]['color'] for v in g.nodes]
        
        g_list.append(S2VGraph(g, 0, node_tags=tags))
    #draw_graph(g, "g1.jpg")
    
    for i in range(number_of_graphs - number_of_outliers, number_of_graphs):
        
        n = np.random.randint(n_min, n_max)
        tag_counts = random_split_counts(n, no_of_tags)

        if type2 == "mixhop":
            g = MixhopGraphGenerator(tag_counts, heteroWeightsExponent=1.0)(n, 5, 10, h_outlier)
            tags = [g.nodes[v]['color'] for v in g.nodes]
        
        g_list.append(S2VGraph(g, 1, node_tags=tags))
    #draw_graph(g, "g2.jpg")
    
    for g in g_list:
        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    if g.node_tags == None:
        print("no node tags provided, using degrees as tags")
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())


    # Extracting unique tags and converting to one-hot features   
    tagset = set()
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

    print('Maximum node tag: %d' % len(tagset))
    print("Number of graphs generated: %d" % (number_of_graphs))
    print("Number of outlier graphs generated: %d" % (number_of_outliers))

    return g_list, 2

def load_synthetic_data_contaminated(number_of_graphs, outlier_ratio=0.5, n_min = 50, n_max = 150, no_of_tags = 2, type1 = "mixhop", type2 = "mixhop"):
    print('generating data')
    g_list = []
    
    number_of_outliers = int(number_of_graphs*outlier_ratio)

    for i in range(number_of_graphs - number_of_outliers):
        
        n = np.random.randint(n_min, n_max)
        tag_counts = random_split_counts(n, no_of_tags)

        if type1 == "mixhop":
            g = MixhopGraphGenerator(tag_counts, heteroWeightsExponent=1.0)(n, 2, 10, 0.5)
            tags = [g.nodes[v]['color'] for v in g.nodes]
        
        g_list.append(S2VGraph(g, 0, node_tags=tags))
    #draw_graph(g, "g1.jpg")
    
    for i in range(number_of_graphs - number_of_outliers, number_of_graphs):
        
        n = np.random.randint(n_min, n_max)
        tag_counts = random_split_counts(n, no_of_tags)

        if type2 == "mixhop":
            g = MixhopGraphGenerator(tag_counts, heteroWeightsExponent=1.0).generate_graph_contaminated(n, 2, 10, 0.5, contamination=1.0)
            tags = [g.nodes[v]['color'] for v in g.nodes]
        
        g_list.append(S2VGraph(g, 1, node_tags=tags))
    #draw_graph(g, "g2.jpg")
    
    for g in g_list:
        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    if g.node_tags == None:
        print("no node tags provided, using degrees as tags")
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())


    # Extracting unique tags and converting to one-hot features   
    tagset = set()
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

    print('Maximum node tag: %d' % len(tagset))
    print("Number of graphs generated: %d" % (number_of_graphs))
    print("Number of outlier graphs generated: %d" % (number_of_outliers))

    return g_list, 2

def load_chem_data(file = "dataset/bace/raw/bace.csv"):
    print('extracting data')
    g_list = []
    
    graphs, labels = extract_chem_graphs(file)
    for g, l in zip(graphs, labels):
        tags = [g.nodes[v]['color'] for v in g.nodes]
        g_list.append(S2VGraph(g, l, node_tags=tags))
    
    for g in g_list:
        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    if g.node_tags == None:
        print("no node tags provided, using degrees as tags")
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())


    # Extracting unique tags and converting to one-hot features   
    tagset = set()
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

    print('Maximum node tag: %d' % len(tagset))
    return g_list, 2


def separate_data(graph_list, seed, fold_idx=0, splits=10):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=splits, shuffle = True, random_state = seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list


G = MixhopGraphGenerator([25,25], heteroWeightsExponent=1.0).generate_graph_contaminated(50, 5, 10, 0.5, contamination=1.0)
draw_graph(G, "uncle")

G = MixhopGraphGenerator([25,25], heteroWeightsExponent=1.0).generate_graph(50, 5, 10, 0.5)
draw_graph(G, "aunty")