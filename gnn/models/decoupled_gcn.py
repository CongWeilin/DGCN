
import torch
import torch.nn as nn

import torch.nn.functional as f
import numpy as np 

class DecoupledGCN(nn.Module):
    """
    New
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
        self.softmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()
        
        # TODO: Change this later to read arch and create corresponding layers
        
        # alpha = 0.2
        alpha = kwargs['alpha']
        TEMP=alpha*(1-alpha)**np.arange(self.num_layers+1)
        TEMP[-1] = (1-alpha)**self.num_layers
        TEMP = np.log(TEMP)
        self.layer_weight = torch.nn.Parameter(torch.tensor(TEMP))
        
        # TEMP = 0.5 /(np.arange(num_layers)+1)
        # TEMP = np.log(TEMP/(1-TEMP))
        # self.identity_map_weight = torch.nn.Parameter(torch.tensor(TEMP))

        self.input_fcs = nn.Linear(features_dim, hidden_dim)
        self.output_fcs = nn.Linear(hidden_dim, num_classes)

        for i in range(num_layers):
            self.layers.append(layer(hidden_dim, hidden_dim, layer_id=i+1))
            
    def forward(self, nodeblocks, x, fetch_hiddens=False):

        # normalize input features
        h = self.batch_norm(x)
        h = self.dropout(x)
        h = self.input_fcs(h)
        h = self.activation(h)
        
        layer_weight = self.softmax(self.layer_weight)      
        
        output_hiddens = h.clone()*layer_weight[0]
        for i, layer in enumerate(self.layers):
            h = layer(nodeblocks[i], h)
            output_hiddens += h.clone()*layer_weight[i+1]
        h = self.dropout(output_hiddens)
        h = self.output_fcs(h)
        
        return h
    

class DecoupledGCN_OGB(nn.Module):
    """
    New
    """

    def __init__(self, features_dim, 
                       hidden_dim, 
                       num_classes, 
                       activation, 
                       layer=None, 
                       arch='', 
                       num_layers=0, 
                       dropout=0, 
                       norm=True,
                       *args, **kwargs):

        super().__init__()

        self.num_layers = num_layers
        self.activation = activation
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = norm

        # TODO: Change this later to read arch and create corresponding layers
        
        self.input_fcs = nn.Linear(features_dim, hidden_dim)
        self.output_fcs = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()
                
        # alpha = 0.2
        alpha = kwargs['alpha']
        TEMP=alpha*(1-alpha)**np.arange(self.num_layers+1)
        TEMP[-1] = (1-alpha)**self.num_layers
        TEMP = np.log(TEMP)
        self.layer_weight = torch.nn.Parameter(torch.tensor(TEMP))
        
        self.identity_map_weight = torch.nn.Parameter(torch.zeros(self.num_layers))
        
        for i in range(self.num_layers):
            self.layers.append(layer(hidden_dim, hidden_dim, layer_id=i+1))

        if norm:
            self.bns = nn.ModuleList()
            for i in range(self.num_layers+1):
                self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
            
    def reset_parameters(self):
        for conv in self.layers:
            conv.reset_parameters()
        
        if self.norm:
            for bn in self.bns:
                bn.reset_parameters()
        self.input_fcs.reset_parameters()
        self.output_fcs.reset_parameters()
        
    def forward(self, nodeblocks, x):
        # normalize input features
        h = self.input_fcs(x)
        if self.norm:
            h = self.bns[0](h)
        h = self.activation(h)
        h = self.dropout(h)
            
        output_hiddens = []
        
        identity_map_weight = self.sigmoid(self.identity_map_weight)
        for i, layer in enumerate(self.layers):
            h, h_w = layer(nodeblocks[i], h, identity_map_weight[i])  
            if self.norm:
                h_w = self.bns[i+1](h_w)
            h_w = self.activation(h_w)
            h_w = self.dropout(h_w)
            output_hiddens.append(h_w)
        
        layer_weight = self.softmax(self.layer_weight)  
        for i in range(len(output_hiddens)):
            output_hiddens[i] = output_hiddens[i] * layer_weight[i]
            
        h = sum(output_hiddens)
        h = self.output_fcs(h)
        
        return h
    
    
# class LocalGCN_OGB_arxiv(nn.Module):
#     """
#     New
#     """

#     def __init__(self, features_dim, hidden_dim, num_classes, activation, layer=None, arch='', num_layers=0, dropout=0, *args, **kwargs):
#         super().__init__()

#         self.num_layers = num_layers
#         self.activation = activation
#         self.layers = nn.ModuleList()
#         self.bns = nn.ModuleList()
#         self.dropout = nn.Dropout(p=dropout)

#         # TODO: Change this later to read arch and create corresponding layers
#         self.layer_weight = torch.nn.Parameter(torch.randn(num_layers))
#         self.input_fcs = nn.Linear(features_dim, hidden_dim)
#         self.output_fcs = nn.Linear(hidden_dim, num_classes)
#         self.softmax = nn.Softmax(dim=0)

#         for i in range(num_layers):
#             self.layers.append(layer(hidden_dim, hidden_dim, layer_id=i+1))
#             self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
            
#         self.reset_parameters()
        
#     def reset_parameters(self):
#         for conv in self.layers:
#             conv.reset_parameters()
#         for bn in self.bns:
#             bn.reset_parameters()
#         self.input_fcs.reset_parameters()
#         self.output_fcs.reset_parameters()
#         torch.nn.init.normal_(self.layer_weight)
        
#     def forward(self, nodeblocks, x):
#         output_hiddens = []
        
#         # normalize input features
#         h = self.input_fcs(x)
#         residual = h
        
#         for i, layer in enumerate(self.layers):
#             h = layer(nodeblocks[i], h)
#             h = self.bns[i](h)
#             h = self.activation(h)
#             h = self.dropout(h)
#             h = h + 0.2 * residual
#             output_hiddens.append(h)
        
#         layer_weight = self.softmax(self.layer_weight)
#         for i in range(len(output_hiddens)):
#             output_hiddens[i] = output_hiddens[i] * layer_weight[i]
            
#         h = sum(output_hiddens)
#         h = self.output_fcs(h)
        
#         return h
    