import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch_sparse import spmm
import math


class GConvII(nn.Module):

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
        
        self.alpha = kwargs['alpha']
        self.beta = kwargs['beta']

        if 'layer_id' in kwargs:
            self.layer_id = kwargs['layer_id']

    def forward(self, adj, h, h0, l):
        theta = math.log(self.beta/l+1)
        hi = adj.spmm(h)
        support = (1 - self.alpha) * hi + self.alpha * h0
        output = theta * self.linear(support) + (1 - theta) * support
        return output

    def __repr__(self):
        return self.__class__.__name__ + "[{}] ({}->{})".format(
            self.layer_id,
            self.input_dim,
            self.output_dim)
