
import torch
import torch.nn as nn

import torch.nn.functional as f

from .base import model_builder


class Custom(nn.Module):
    """Custom GNN model builder

    Arguments:
        nn {[type]} -- [description]
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
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation
        self.batch_norm = nn.BatchNorm1d(features_dim, affine=False)
        self.arch = arch

        prop_layer = kwargs['prop']
        kwargs.pop('prop')
        self.layers = model_builder(
            arch, layer, prop_layer,
            features_dim, hidden_dim, num_classes,
            *args, **kwargs)

    def forward(self, nodeblocks, x):

        # normalize input features
        h = self.batch_norm(x)

        i = 0
        for layer in self.layers:
            if i > 0:
                    h = self.dropout(h)

            if isinstance(layer, nn.Linear):
                h = layer(h)
            else:
                h = layer(nodeblocks[i], h)
                i += 1

            if i < self.num_layers - 1:
                h = self.activation(h)

        return h
