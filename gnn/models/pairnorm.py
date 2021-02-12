
import torch
import torch.nn as nn

import torch.nn.functional as f

from ..layers import PairNorm

class GCN_Pairnorm(nn.Module):
    """
    GCN model with simple GConv layers at all layers
    """

    def __init__(self,
                 features_dim,
                 hidden_dim,
                 num_classes,
                 activation,
                 layer=None,
                 arch='',
                 num_layers=0,
                 dropout=0,
                 *args,
                 **kwargs):

        super().__init__()

        self.num_layers = num_layers
        self.activation = activation
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        self.pairnorm = PairNorm(mode='PN', scale=1)
        self.batch_norm = nn.BatchNorm1d(features_dim, affine=False)

        # TODO: Change this later to read arch and create corresponding layers
        self.layers.append(layer(features_dim, hidden_dim, layer_id=1))
        for i in range(1, num_layers-1):
            self.layers.append(layer(hidden_dim, hidden_dim, layer_id=i+1))
        self.layers.append(layer(hidden_dim, num_classes, layer_id=num_layers))

    def forward(self, nodeblocks, x):

        # normalize input features
        h = self.batch_norm(x)

        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(nodeblocks[i], h)
            h = self.pairnorm(h)
            if i < self.num_layers - 1:
                h = self.activation(h)
        return h
