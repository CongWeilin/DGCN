
import torch
import torch.nn as nn

import torch.nn.functional as f

from .base import model_builder


class GCN(nn.Module):
    """
    General GCN with options for batch norm for input features and per layer batch norm
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
                 input_norm=True,
                 layer_norm=False,
                 *args,
                 **kwargs):

        super().__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation

        self.input_norm = input_norm
        self.layer_norm = layer_norm

        if input_norm:
            self.input_batchnorm = nn.BatchNorm1d(features_dim, affine=False)
        
        self.layers.append(layer(features_dim, hidden_dim, layer_id=1))
        for i in range(1, num_layers-1):
            self.layers.append(layer(hidden_dim, hidden_dim, layer_id=i+1))
        self.layers.append(layer(hidden_dim, num_classes, layer_id=num_layers))

        if layer_norm:
            self.layer_batchnorms = nn.ModuleList()
            for _ in range(num_layers):
                self.layer_batchnorms.append(torch.nn.BatchNorm1d(hidden_dim))

    def forward(self, nodeblocks, x):

        # normalize input features
        h = self.input_batchnorm(x) if self.input_norm else x

        for i, layer in enumerate(self.layers):
            # h = self.dropout(h) # Removed dropout
            h = layer(nodeblocks[i], h)
            if i < self.num_layers - 1:
                if self.layer_norm:
                    h = self.layer_batchnorms[i](h)
                h = self.activation(h)
                h = self.dropout(h)

        return h

class GCN_Paper(nn.Module):
    """
    GCN implementation from Kipf et. al paper
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
                 input_norm=True,
                 *args,
                 **kwargs):

        super().__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation

        self.input_norm = input_norm
        if input_norm:
            self.batch_norm = nn.BatchNorm1d(features_dim, affine=False)

        self.layers.append(layer(features_dim, hidden_dim, layer_id=1))
        for i in range(1, num_layers-1):
            self.layers.append(layer(hidden_dim, hidden_dim, layer_id=i+1))
        self.layers.append(
            layer(hidden_dim, num_classes, layer_id=num_layers))

    def forward(self, nodeblocks, x):

        # normalize input features
        if self.input_norm:
            h = self.batch_norm(x)
        else:
            h = x

        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(nodeblocks[i], h)
            if i < self.num_layers - 1:
                h = self.activation(h)

        return h

class GCN_OGB(nn.Module):
    """
    GCN model with BatchNorm at all layers from ogb leaderboard
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
        self.bns = nn.ModuleList()

        self.dropout = nn.Dropout(p=dropout)

        self.layers.append(layer(features_dim, hidden_dim, layer_id=1))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        for i in range(1, num_layers-1):
            self.layers.append(layer(hidden_dim, hidden_dim, layer_id=i+1))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        self.layers.append(layer(hidden_dim, num_classes, layer_id=num_layers))

    def forward(self, nodeblocks, x):

        # normalize input features
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(nodeblocks[i], h)
            if i < self.num_layers - 1:
                h = self.bns[i](h)
                h = self.activation(h)
                h = self.dropout(h)
        return h
