import torch
from typing import Tuple

from torch import nn

from DiGCN_Inception_Block import DiGCN_InceptionBlock as InceptionBlock
import torch.nn.functional as F
import dgl.nn as dglnn

class DiGCN_Inception_Block_Ranking(nn.Module):
    r"""The ranking model adapted from the
    `Digraph Inception Convolutional Networks"
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.
    Args:
        * **num_features** (int): Number of features.
        * **dropout** (float): Dropout probability.
        * **embedding_dim** (int) - Embedding dimension.
        * **Fiedler_layer_num** (int, optional) - The number of single Filder calculation layers, default 3.
        * **alpha** (float, optional) - (Initial) learning rate for the Fiedler step, default 0.01.
        * **trainable_alpha** (bool, optional) - Whether alpha is trainable, default False.
        * **initial_score** (torch.FloatTensor, optional) - Initial guess of scores, default None.
        * **sigma** (float, optionial) - (Initial) Sigma in the Gaussian kernel, actual sigma is this times sqrt(num_nodes), default 1.
        * **kwargs (optional): Additional arguments of
            :class:`RankingGNNBase`.
    """

    def __init__(self, num_features: int, dropout: float, embedding_dim: int,out:int):
        super().__init__()
        self.ib1 = InceptionBlock(num_features, embedding_dim)
        self.ib2 = InceptionBlock(embedding_dim, out)
        self._dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.ib1.reset_parameters()
        self.ib2.reset_parameters()

    def forward(self,edge_index_tuple: Tuple[torch.LongTensor, torch.LongTensor],
                edge_weight_tuple: Tuple[torch.FloatTensor, torch.FloatTensor],
                features: torch.FloatTensor, ) -> torch.FloatTensor:
        """
        Making a forward pass.
        Arg types:
            * edge_index_tuple (PyTorch LongTensor) - Tuple of edge indices.
            * edge_weight_tuple (PyTorch FloatTensor, optional) - Tuple of edge weights corresponding to edge indices.
            * features (PyTorch FloatTensor) - Node features.

        Return types:
            * z (PyTorch FloatTensor) - Embedding matrix.
        """
        x = features
        edge_index, edge_index2 = edge_index_tuple
        edge_index, edge_index2 = edge_index.reshape(1,edge_index.size(0)), edge_index.reshape(1,edge_index2.size(0))
        edge_index, edge_index2 = torch.cat((edge_index,edge_index2), 0),torch.cat((edge_index,edge_index2), 0)
        edge_weight, edge_weight2 = edge_weight_tuple
        x0, x1, x2 = self.ib1(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self._dropout)
        x1 = F.dropout(x1, p=self._dropout)
        x2 = F.dropout(x2, p=self._dropout)
        x = x0.type(torch.DoubleTensor) + x1.type(torch.DoubleTensor) + x2.type(torch.DoubleTensor)
        x = F.dropout(x, p=self._dropout)

        x0, x1, x2 = self.ib2(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self._dropout)
        x1 = F.dropout(x1, p=self._dropout)
        x2 = F.dropout(x2, p=self._dropout)
        x = x0 + x1.type(torch.DoubleTensor) + x2.type(torch.DoubleTensor)
        self.z = x.clone()

        return self.z

class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h
