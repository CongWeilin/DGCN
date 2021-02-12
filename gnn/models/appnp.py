
import torch
import torch.nn as nn

import torch.nn.functional as f

class APPNP_Paper(nn.Module):
    """
    APPNP from paper
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

        self.batch_norm = nn.BatchNorm1d(features_dim, affine=False)

        # TODO: Change this later to read arch and create corresponding layers
        self.input_fcs = nn.Linear(features_dim, hidden_dim)
        self.output_fcs = nn.Linear(hidden_dim, num_classes)

        for i in range(num_layers):
            self.layers.append(layer(hidden_dim, hidden_dim, alpha=0.2, layer_id=i+1))

    def forward(self, nodeblocks, x):

        # normalize input features
        h = self.batch_norm(x)

        h = self.dropout(x)
        h = self.input_fcs(h)
        h = self.activation(h)

        h_0 = h.clone()
        
        for i, layer in enumerate(self.layers):
            h = layer(nodeblocks[i], h, h_0)
        
        h = self.dropout(h)
        h = self.output_fcs(h)
        
        return h
    
class APPNP_OGB(nn.Module):
    """
    APPNP from leaderboard
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

        # normalize input features
        h = self.dropout(x)
        h = self.input_fcs(h)
        # h = self.bns[0](h)
        h = self.activation(h)

        h_0 = h.clone()
        
        for i, layer in enumerate(self.layers):
            h = layer(nodeblocks[i], h, h_0)
            h = self.bns[i](h)
        
        # h = self.bns[1](h)
        h = self.dropout(h)
        h = self.output_fcs(h)
        
        return h
    
    