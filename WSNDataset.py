import dgl
import numpy as np
import pandas as pd
import torch
from dgl import save_graphs
from dgl.data import DGLDataset


class WSNDataset(DGLDataset):
    def __init__(self):
        self.edges_data = pd.read_csv("edges.csv")
        self.nodes_data = pd.read_csv('./features.csv')
        self.preprocess_edges()
        super().__init__(name='wsn_data')

    def preprocess_edges(self):
        list_dst = self.nodes_data[' id']
        self.edges_data['remove'] = [True if x not in list_dst else False for x in self.edges_data['Dst']]
        self.edges_data = self.edges_data[self.edges_data['remove']!=True]
        self.edges_data.drop(columns=["remove"],inplace=True)

    def process(self):
        nodes_data = pd.read_csv('./features.csv')
        a = pd.read_csv("label.csv")
        print((a[a['Attack type'] == "Normal"]).shape)
        node_labels = torch.from_numpy(a['Attack type'].astype("category").cat.codes.to_numpy()).type(
            torch.FloatTensor)
        # nodes_data.reset_index(inplace=True)
        new_Dst = []
        for i in self.edges_data['Dst']:
            try:
                new_Dst.append(nodes_data[nodes_data[' id'] == i].index.values[0])
            except:
                new_Dst.append(np.nan)
        self.edges_data['Dst'] = new_Dst
        self.edges_data.dropna(how="any",inplace=True)
        self.edges_data['Src'] = [nodes_data[nodes_data[' id'] == x].index.values[0] for x in self.edges_data['Src']]
        nodes_data.drop(columns=[' id'], inplace=True)
        node_features = torch.from_numpy(nodes_data.to_numpy()).type(torch.FloatTensor)
        edge_features = torch.from_numpy(self.edges_data['Weight'].to_numpy())
        edges_src = torch.from_numpy(self.edges_data['Src'].to_numpy(dtype=int))
        edges_dst = torch.from_numpy(self.edges_data['Dst'].to_numpy(dtype=int))

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.ndata["label"] = node_labels
        self.graph.edata['weight'] = edge_features

        n_nodes = nodes_data.shape[0]
        n_train = int(n_nodes * 0.6)
        # n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        # val_mask[n_train: n_train + n_val] = True
        # test_mask[n_train + n_val:] = True
        test_mask[n_train:] = True
        self.graph.ndata["train_mask"] = train_mask
        # self.graph.ndata["val_mask"] = val_mask
        self.graph.ndata["test_mask"] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


dataset = WSNDataset()
graph = dataset[0]
save_graphs("./data.bin", [graph])
