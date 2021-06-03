import numpy as np
import scipy.sparse as sp
from utils import sparse_mx_to_torch_sparse_tensor

import multiprocessing as mp
import time
import os
import torch

class DataLoader(object):
    def __init__(self, adj_mat, train_nodes, valid_nodes, test_nodes, device):
        self.adj_mat = adj_mat
        self.train_nodes = train_nodes
        self.valid_nodes = valid_nodes
        self.test_nodes = test_nodes
        self.device = device
        self.num_nodes = adj_mat.shape[0]
        self.num_train_nodes = len(self.train_nodes)
        self.lap_matrix = self.sym_normalize(adj_mat)
        self.lap_tensor = sparse_mx_to_torch_sparse_tensor(self.lap_matrix)
        self.lap_tensor = torch.sparse.FloatTensor(self.lap_tensor[0],
                                                   self.lap_tensor[1], 
                                                   self.lap_tensor[2]).to(device)

    def get_mini_batches(self, batch_size):
        train_nodes = np.random.permutation(self.train_nodes)
        start = 0
        end = batch_size
        mini_batches = []
        while True:
            mini_batches.append(train_nodes[start:end])
            start = end
            end = start + batch_size
            if end > self.num_train_nodes:
                break
        return mini_batches, self.lap_tensor
            
    def get_train_batch(self, ):
        return self.train_nodes, self.lap_tensor

    def get_valid_batch(self,):
        return self.valid_nodes, self.lap_tensor

    def get_test_batch(self,):
        return self.test_nodes, self.lap_tensor
    
    def sym_normalize(self, adj):
        """Normalization by D^{-1/2} (A+I) D^{-1/2}."""
        adj.setdiag(1)
        rowsum = np.array(adj.sum(1)) + 1e-20
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)
        adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        return adj

    def get_mini_batch_dropedge(self, percent=0.8):
        nnz = self.adj_mat.nnz
        perm = np.random.permutation(nnz)
        preserve_nnz = int(nnz*percent)
        perm = np.sort(perm[:preserve_nnz])
        # print(preserve_nnz, perm)
        adj_mat = self.adj_mat.tocoo()
        adj_mat = sp.coo_matrix((adj_mat.data[perm],
                               (adj_mat.row[perm],
                                adj_mat.col[perm])),
                              shape=adj_mat.shape)
        lap_matrix = self.sym_normalize(adj_mat)
        lap_tensor = sparse_mx_to_torch_sparse_tensor(lap_matrix)
        lap_tensor = torch.sparse.FloatTensor(lap_tensor[0],
                                              lap_tensor[1], 
                                              lap_tensor[2]).to(self.device)
        return [self.train_nodes], lap_tensor