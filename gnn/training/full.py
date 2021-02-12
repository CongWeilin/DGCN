import copy
import time
import torch
import torch.nn.functional as f
import torch_geometric.transforms as T
from sklearn.preprocessing import StandardScaler

from . import Base
from ..data import NodeBlocks, Graph
from ..data import transforms as GT
from ..data.graph import graph_normalize


class Full(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_init = copy.deepcopy(self.model)

        # Prepare features and move to device
        # ! this needs change if features don't fit
        self.x = self.data.x.to(self.device)
        self.y = self.data.y.squeeze().to(self.device)

        # prepare nodeblock and normalize it and move to device
    
        # FIXME: dirty hack
        # replace this with preprocess
        if self.dataset.name.startswith('ogbn'):
            self.data.adj_t = self.data.adj_t.to_symmetric()
        if self.config.fully_connected:
            self.data.adj_t = GT.FullyConnected()(self.data)

        # Normalize the full graph
        graph = graph_normalize(self.config.adj_norm, self.data.adj_t)

        # Full graph nodeblock, used for train,val and test
        # in case of batch-gcn, we might still need this for
        # validation and test, hence done in __init__
        self.nbs = NodeBlocks(self.config.num_layers)
        self.nbs.from_graph(graph)
        self.nbs.to(self.device)

    def run(self):

        self.train_nbs = self.nbs

        for epoch in self.tbar:

            # import pdb; pdb.set_trace()
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

    def clip_grads(self, clip_value=0.2):
        for p in self.model.parameters():
            torch.nn.utils.clip_grad_norm_(p, clip_value)

    def train(self, epoch):

        # Start Time for Epoch
        torch.cuda.synchronize()
        train_start = time.perf_counter()

        self.model.train()
        self.optimizer.zero_grad()

        # Forward
        pred = self.model(self.train_nbs, self.x)

        # Loss
        loss = self.loss_func(
            pred[self.data.train_mask], self.y[self.data.train_mask])

        # Backward
        loss.backward()
        # self.clip_grads()
        self.optimizer.step()

        # End Time for Epoch
        torch.cuda.synchronize()
        train_end = time.perf_counter()
        self.stats.train_time_epochs.append(train_end - train_start)

        # update stats
        train_score = self.calc_f1(
            pred[self.data.train_mask], self.y[self.data.train_mask])
        self.stats.train_loss.append(loss.item())
        self.stats.train_scores.append(train_score)

    @torch.no_grad()
    def validation(self, epoch):
        
        if epoch % self.config.val_steps != 0:
            return
            
        self.model.eval()
        pred = self.model(self.nbs, self.x)[self.data.val_mask]
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
        self.stats.best_model.eval()
        pred = self.stats.best_model(self.nbs, self.x)[self.data.test_mask]
        labels = self.y[self.data.test_mask]
        test_loss = self.loss_func(pred, labels).item()
        test_score = self.calc_f1(pred, labels)
        self.stats.test_score = test_score


    def post_epoch(self, ):

        # compute weight norms of best model
        weight_norms = dict()
        for n, p in self.stats.best_model.named_parameters():
            weight_norms[n] = p.data.norm(2).item()
        self.stats.weight_norms = weight_norms

        # compute gradient of un-trained model
        self.model_init.to(self.device)
        self.model_init.zero_grad()
        pred = self.model_init(self.nbs, self.x)
        loss = self.loss_func(
            pred[self.data.train_mask], self.y[self.data.train_mask])
        loss.backward()
        grads_norms = dict()
        for n, p in self.model.named_parameters():
            grads_norms[n] = p.grad.data.norm(2).item()
        self.model_init.zero_grad()
        self.stats.grads_norms = grads_norms

        
    @torch.no_grad()
    def fetch_hiddens(self, inner_layers=False):
        self.stats.best_model.eval()
        hiddens = self.stats.best_model(self.nbs, self.x, fetch_hiddens=inner_layers)
        return hiddens


    def graph_preprocess(self, adj_t):
        pass