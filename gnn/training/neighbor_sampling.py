import copy
import torch
import torch.nn.functional as f
import torch_geometric.transforms as T
from sklearn.preprocessing import StandardScaler

from . import Full
from ..data import NodeBlocks, Graph
from ..data import transforms as GT
from ..data.graph import graph_normalize


class NeighborSampling(Full):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        
        for epoch in self.tbar:
            
            # create nodeblocks
            self.train_nbs = self.create_nodeblocks()

            # move to device
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

    
    def create_nodeblocks(self):
        nodeblocks = NodeBlocks(self.config.num_layers)
        for l in range(self.config.num_layers):
            layer_graph, layer_nodes = self.sample_neighbors(self.data.adj_t)
            norm_layer_graph = graph_normalize(self.config.adj_norm, layer_graph)
            nodeblocks.add_layers(norm_layer_graph, layer_nodes)
        return nodeblocks

    def sample_neighbors(self, adj):
        """ Sample from adj from the neighbors

        Arguments:
            adj {[type]} -- [description]

        Returns:
            [type] -- [description]
        """        
        node_idx = torch.arange(adj.sizes()[0])
        sampled_adj, _ = adj.sample_adj(subset=node_idx, num_neighbors=self.config.num_neighbors)

        return sampled_adj, node_idx

    @torch.no_grad()
    def validation(self, epoch):
        
        if self.config.exact_val:
            return super().validation(epoch)

        self.model.eval()

        val_nodeblocks = self.create_nodeblocks()
        val_nodeblocks.to(self.device)

        pred = self.model(val_nodeblocks, self.x)[self.data.val_mask]
        labels = self.y[self.data.val_mask]
        val_loss = self.loss_func(pred, labels).item()
        val_score = self.calc_f1(pred, labels)

        if val_score > self.stats.best_val_score:
            self.stats.best_val_epoch = epoch
            self.stats.best_val_loss = val_loss
            self.stats.best_val_score = val_score
            self.stats.best_model = copy.deepcopy(self.model)

        self.stats.val_loss.append(val_loss)
        self.stats.val_scores.append(val_score)
        

    @torch.no_grad()
    def inference(self):
        
        if self.config.exact_infer:
            return super().inference()

        self.stats.best_model.eval()
        infer_nodeblocks =  self.create_nodeblocks()
        infer_nodeblocks.to(self.device)

        pred = self.stats.best_model(infer_nodeblocks, self.x)[self.data.test_mask]
        labels = self.y[self.data.test_mask]
        test_loss = self.loss_func(pred, labels).item()
        test_score = self.calc_f1(pred, labels)
        self.stats.test_score = test_score