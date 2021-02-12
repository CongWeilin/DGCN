import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import spmm
import math
from .utils import glorot, zeros
    
# class LocalGConv(nn.Module):

#     def __init__(self,
#                  input_dim,
#                  output_dim,
#                  step_length,
#                  *args,
#                  **kwargs,
#                  ):
#         super().__init__()

#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.linear = nn.Linear(input_dim, output_dim)
        
#         self.step_length = step_length
#         self.layer_weight = torch.nn.Parameter(torch.randn(step_length+1))
#         self.softmax = nn.Softmax(dim=0)
        
#         if 'layer_id' in kwargs:
#             self.layer_id = kwargs['layer_id']

#     def forward(self, adj, h):
#         output_hiddens = [h]
#         for step in range(self.step_length):
#             h = adj.spmm(h)
#             output_hiddens.append(h)
            
#         layer_weight = self.softmax(self.layer_weight)
#         for i in range(len(output_hiddens)):
#             output_hiddens[i] = output_hiddens[i] * layer_weight[i]
            
#         h = sum(output_hiddens)
#         h = self.linear(h)
#         return h

#     def __repr__(self):
#         return self.__class__.__name__ + "[{}] ({}->{})".format(
#             self.layer_id,
#             self.input_dim,
#             self.output_dim)
    
#     def reset_parameters(self):
#         glorot(self.linear.weight)
#         zeros(self.linear.bias)
#         torch.nn.init.normal_(self.layer_weight)
    
class LocalGConv(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 *args,
                 **kwargs,
                 ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)
        # self.eps = torch.nn.Parameter(torch.tensor([1e-10]))
        
        if 'layer_id' in kwargs:
            self.layer_id = kwargs['layer_id']

    def forward(self, adj, h, identity_map_weight):
        h = adj.spmm(h) 
        h_w = identity_map_weight * self.linear(h) + (1-identity_map_weight) * h
        return h, h_w

    def __repr__(self):
        return self.__class__.__name__ + "[{}] ({}->{})".format(
            self.layer_id,
            self.input_dim,
            self.output_dim)
    
    def reset_parameters(self):
        glorot(self.linear.weight)
        zeros(self.linear.bias)