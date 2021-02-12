import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import to_scipy_sparse_matrix
import torch.nn.functional as f
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import numpy as np

from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops, remove_self_loops

from . import transforms as T

class Graph(object):

    def __init__(self, data, normalized=None):

        self.adj: SparseTensor = None
        self.num_nodes = data.num_nodes
        self.num_edges = data.num_edges  # doesn't count self loops!

        # FIXME: maybe labels are different?
        self.nodes = torch.arange(self.num_nodes)

        edge_index = data.edge_index.cpu()
        edge_index, _ = add_self_loops(edge_index)
        edge_val = torch.ones(edge_index.size()[1])

        self.adj = SparseTensor(row=edge_index[0],
                                col=edge_index[1],
                                value=edge_val,
                                sparse_sizes=(self.num_nodes, self.num_nodes),
                                is_sorted=False)
        self.E_mat = self.compute_subspace_projection(self, edge_index)
    
    def compute_subspace_projection(self, edge_index):
        row, col = edge_index
        data = np.ones(row)
        graph = sp.csr_matrix((data, (row, col)), shape=(self.num_nodes, self.num_nodes))
        degree = np.array(adj.sum(1)).flatten()
        n_components, labels = connected_components(csgraph=graph, 
                                                    directed=False, return_labels=True)
        
        E = np.zeros(n_components, self.num_nodes)
        for i in range(n_components):
            e = degree[labels==i]
            e = np.sqrt(e)/np.sqrt(np.sum(e))
            E[i, labels==i] = e
        return E
            

    @property
    def laplacian(self):
        return NotImplementedError

def graph_normalize(norm_mode, adj_t):
    if norm_mode == 'row':
        graph = T.RowNorm()(adj_t)
    elif norm_mode == 'col':
        graph = T.ColNorm()(adj_t)
    elif norm_mode == 'sym':
        graph = T.SymNorm()(adj_t)
    else:
        raise ValueError('Unknown normalization!')
    
    return graph