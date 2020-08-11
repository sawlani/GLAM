from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
import pandas as pd

def mol_to_nx(mol):
    """
    Converts rdkit mol object to NetworkX graph
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    G = nx.Graph()
    #labels = {}
    for atom in mol.GetAtoms():
        v = atom.GetIdx()
        G.add_node(v, color=atom.GetAtomicNum())
        #labels[v] = atom.GetSymbol()

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        wt = 1
        bondtype = bond.GetBondType()
        if bondtype == Chem.rdchem.BondType.DOUBLE:
            wt = 2
        if bondtype == Chem.rdchem.BondType.TRIPLE:
            wt = 3
        
        G.add_edge(i, j, weight=wt)

    return G


def extract_chem_graphs(filename="smiles.csv", max_graphs = 200):
    graphs = []
    labels = []

    df = pd.read_csv(filename).head(max_graphs)
    smiles = df['mol'].tolist()
    labels = df['Class'].tolist()

    graphs = [mol_to_nx(AllChem.MolFromSmiles(smile)) for smile in smiles]
    
    return graphs, labels
