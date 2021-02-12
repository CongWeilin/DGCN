
import torch
import torch.nn as nn

import torch.nn.functional as f

class GCNII(nn.Module):
    """
    GCNII from https://github.com/chennnM/GCNII/blob/ca91f5686c4cd09cc1c6f98431a5d5b7e36acc92/model.py#L41
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

        self.input_fcs = nn.Linear(features_dim, hidden_dim)
        self.output_fcs = nn.Linear(hidden_dim, num_classes)

        for i in range(num_layers):
            self.layers.append(layer(hidden_dim, hidden_dim, layer_id=i+1, *args, **kwargs))

    def forward(self, nodeblocks, x, fetch_hiddens=False):

        # normalize input features
        if self.input_norm:
            h = self.batch_norm(x)
        else:
            h = x

        h = self.dropout(x)
        h = self.input_fcs(h)
        h = self.activation(h)

        h_0 = h.clone()
        
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(nodeblocks[i], h, h_0, i+1)
            h = self.activation(h)
        
        h = self.dropout(h)
        h = self.output_fcs(h)
        
        return h
    
class GCNII_OGB(nn.Module):
    """
    https://github.com/chennnM/GCNII/blob/master/PyG/ogbn-arxiv/arxiv.py
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

        # TODO: Change this later to read arch and create corresponding layers
        self.input_fcs = nn.Linear(features_dim, hidden_dim)
        self.output_fcs = nn.Linear(hidden_dim, num_classes)

        for i in range(num_layers):
            self.layers.append(layer(hidden_dim, hidden_dim, layer_id=i+1, *args, **kwargs))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
    
    def forward(self, nodeblocks, x, fetch_hiddens=False):

        # normalize input features
        h = self.input_fcs(x)
        h = self.activation(h)
        
        h_0 = h.clone()
        
        for i, layer in enumerate(self.layers):
            h = layer(nodeblocks[i], h, h_0, i+1)
            h = self.bns[i](h)
            h = self.activation(h)
            h = self.dropout(h)
        
        h = self.output_fcs(h)
        return h

class GCNII_Arxiv(nn.Module):
    """
    !INCOMPLETE, DON'T use
    Adapted from original code here
    https://github.com/chennnM/GCNII/blob/master/PyG/ogbn-arxiv/arxiv.py
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

        # TODO: Change this later to read arch and create corresponding layers
        self.input_fcs = nn.Linear(features_dim, hidden_dim)
        self.output_fcs = nn.Linear(hidden_dim, num_classes)

        for i in range(num_layers):
            self.layers.append(layer(hidden_dim, hidden_dim, layer_id=i+1, *args, **kwargs))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
    
    def forward(self, nodeblocks, x, fetch_hiddens=False):

        # normalize input features
        h = self.input_fcs(x)
        h = self.activation(h)
        
        h_0 = h.clone()
        
        for i, layer in enumerate(self.layers):
            h = layer(nodeblocks[i], h, h_0, i+1)
            h = self.bns[i](h)
            h = self.activation(h)
            h = self.dropout(h)
        
        h = self.output_fcs(h)
        return h