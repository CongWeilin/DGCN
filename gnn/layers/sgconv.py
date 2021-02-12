import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import spmm
import math


class SGConv(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 *args,
                 **kwargs,
                 ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if 'layer_id' in kwargs:
            self.layer_id = kwargs['layer_id']

    def forward(self, adj, h):

        h = adj.spmm(h)
        return h

    def __repr__(self):
        return self.__class__.__name__ + "[{}] ({}->{})".format(
            self.layer_id,
            self.input_dim,
            self.output_dim)
