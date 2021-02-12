
import torch
import torch.nn as nn

import torch.nn.functional as f

class ResGCN(nn.Module):
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
                 input_norm=True,
                 *args,
                 **kwargs):

        super().__init__()

        self.num_layers = num_layers
        self.activation = activation
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)

        self.input_norm = input_norm
        if input_norm:
            self.batch_norm = nn.BatchNorm1d(features_dim, affine=False)

        # TODO: Change this later to read arch and create corresponding layers
        self.layers.append(layer(features_dim, hidden_dim, layer_id=1))
        for i in range(1, num_layers-1):
            self.layers.append(layer(hidden_dim, hidden_dim, layer_id=i+1))
        self.layers.append(layer(hidden_dim, num_classes, layer_id=num_layers))

    def forward(self, nodeblocks, x):

        # normalize input features
        if self.input_norm:
            h = self.batch_norm(x)
        else:
            h = x
            
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            if i > 0:
                h_res = h.clone()
                h = layer(nodeblocks[i], h)
                if i < self.num_layers - 1:
                    h = self.activation(h) + h_res
            else:
                h = layer(nodeblocks[i], h)
                h = self.activation(h)
        return h


class ResGCN_OGB(nn.Module):
    """
    Residual GCN tuned for OGB
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

        self.bns = nn.ModuleList()

        self.input_fcs = nn.Linear(features_dim, hidden_dim)
        self.output_fcs = nn.Linear(hidden_dim, num_classes)

        for i in range(num_layers):
            self.layers.append(layer(hidden_dim, hidden_dim, layer_id=i+1, *args, **kwargs))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

    def forward(self, nodeblocks, x):
        
        h = self.input_fcs(x)

        for i, layer in enumerate(self.layers):
            h_res = h.clone()
            h = layer(nodeblocks[i], h) 
            h = self.bns[i](h)
            h = self.activation(h)
            h = self.dropout(h)
            h = h + h_res

        h = self.output_fcs(h)
        
        return h