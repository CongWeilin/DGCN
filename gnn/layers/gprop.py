import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import spmm
import math


class GProp(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 *args,
                 **kwargs,
                 ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.alpha = kwargs['alpha']

        if 'layer_id' in kwargs:
            self.layer_id = kwargs['layer_id']

    def forward(self, adj, h, h0, *args):

        h = adj.spmm(h)

        # support = (1 - self.alpha) * h + self.alpha * h0

        return h
        # return support

    def __repr__(self):
        return self.__class__.__name__ + "[{}] ({}->{})".format(
            self.layer_id,
            self.input_dim,
            self.output_dim)
