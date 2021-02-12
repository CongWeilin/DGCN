import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import spmm
from .utils import glorot, zeros

class GConv(nn.Module):

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

        if 'layer_id' in kwargs:
            self.layer_id = kwargs['layer_id']

    def forward(self, adj, h, *args):

        h = adj.spmm(h)
        h = self.linear(h)
        return h

    def __repr__(self):
        return self.__class__.__name__ + "[{}] ({}->{})".format(
            self.layer_id,
            self.input_dim,
            self.output_dim)
    
    def reset_parameters(self):
        glorot(self.linear.weight)
        zeros(self.linear.bias)
        