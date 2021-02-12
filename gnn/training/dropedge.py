import torch
import numpy as np
from torch_sparse import SparseTensor

from . import Full
from ..data import NodeBlocks
from ..data.graph import graph_normalize


class DropEdge(Full):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):

        for epoch in self.tbar:

            # prepare self.nbs and move
            graph = self.dropedge(self.data.adj_t)
            graph = graph_normalize(self.config.adj_norm, graph)

            self.train_nbs = NodeBlocks(self.config.num_layers)
            self.train_nbs.from_graph(graph)
            self.train_nbs.to(self.device)

            self.train(epoch)
            self.validation(epoch)

            # update bar
            self.tbar.set_postfix(loss=self.stats.train_loss[-1],
                                  epoch_score=self.stats.train_scores[-1],
                                  val_loss=self.stats.val_loss[-1],
                                  val_score=self.stats.val_scores[-1])

        self.inference()

        print('Test Score: {:.2f}% @ Epoch #{}'.format(
            self.stats.test_score * 100,
            self.stats.best_val_epoch)
        )

    def dropedge(self, adj):

        # Old way, similar to GraphGuess, not exactly as DropEdge Paper
        # node_idx = torch.arange(adj.sizes()[0])
        # de_adj, _ = adj.sample_adj(subset=node_idx, num_neighbors=self.config.num_neighbors)

        # old fix for pytorch_sparse bug
        # ts = de_adj.to_torch_sparse_coo_tensor()
        # de_adj = SparseTensor.from_torch_sparse_coo_tensor(ts)

        # New Way, probably not as efficient
        adj_scipy = adj.to_scipy(layout='csr')
        num_edges = adj_scipy.data.shape[0]
        num_droped_edges = int(num_edges * self.config.drop_ratio)
        droped_indices = np.random.choice(
            np.arange(num_edges), num_droped_edges)

        adj_scipy.data[droped_indices] = 0
        adj_scipy.eliminate_zeros()

        de_adj = SparseTensor.from_scipy(adj_scipy)

        return de_adj
