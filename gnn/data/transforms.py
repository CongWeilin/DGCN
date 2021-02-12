import torch

from torch_geometric.nn.conv.gcn_conv import gcn_norm

from torch_geometric.utils import add_self_loops
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul, coalesce
from torch_scatter import scatter_add



class RowNorm(object):
    def __call__(self, adj):
        if isinstance(adj, SparseTensor):
            # Add self loop
            adj_t = fill_diag(adj, 1)

            deg = sum(adj_t, dim=1)
            deg_inv = 1. / deg
            deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
            adj_t = mul(adj_t, deg_inv.view(-1, 1))
            return adj_t


class ColNorm(object):
    def __call__(self, adj):
        if isinstance(adj, SparseTensor):
            # Add self loop
            adj_t = fill_diag(adj, 1)

            deg = sum(adj_t, dim=0)
            deg_inv = 1. / deg
            deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
            adj_t = mul(adj_t, deg_inv.view(-1, 1))
            return adj_t

class SymNorm(object):
    def __call__(self, adj):
        if isinstance(adj, SparseTensor):
            adj_t = gcn_norm(adj)
            return adj_t

class FullyConnected(object):
    def __call__(self, data):
        # Worst Idea ever!
        dense_full = torch.ones(data.num_nodes, data.num_nodes)
        adj_t = SparseTensor.from_dense(dense_full)
        return adj_t

