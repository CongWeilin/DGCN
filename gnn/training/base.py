import torch
import torch.nn as nn
import torch.optim as optim
# from tqdm.autonotebook import tqdm, trange
from tqdm import trange

import copy
from sklearn.metrics import f1_score, accuracy_score
from ogb.nodeproppred import Evaluator

from ..utils import Stats
from ..models import *
from ..layers import *
from ..datasets import *


class Base(object):
    def __init__(self,
                 config,
                 dataset,
                 *args,
                 **kwargs
                 ):

        self.dataset = dataset
        # for easier access to adj

        # Custom split of dataset
        # print(config.create_split)
        if config.create_split == '60/20/20':
            data = dataset[0]
            train_rate, valid_rate = 0.6, 0.2
            percls_train_num = int(
                round(train_rate*len(data.y)/dataset.num_classes))
            valid_num = int(round(valid_rate*len(data.y)))
            self.data = random_splits(
                data, dataset.num_classes, percls_train_num, valid_num)
            print('Create split with per class train num %d and valid num %d' %
                  (percls_train_num, valid_num))
        elif config.create_split == '5/5/90':
            data = dataset[0]
            train_rate, valid_rate = 0.05, 0.05
            percls_train_num = int(
                round(train_rate*len(data.y)/dataset.num_classes))
            valid_num = int(round(valid_rate*len(data.y)))
            self.data = random_splits(
                data, dataset.num_classes, percls_train_num, valid_num)
            print('Create split with per class train num %d and valid num %d' %
                  (percls_train_num, valid_num))
        else:
            print('Default split')
            self.data = dataset[0]

        self.config = config

        # Converting config to actual classes/functions
        layer = layer_selector(self.config.layer)
        model = model_selector(self.config.model)

        # Selecting Activation Function
        if self.config.activation == 'relu':
            activation = nn.ReLU(True)
        else:
            raise NotImplementedError

        prop = layer_selector(self.config.prop)
        kwargs['prop'] = prop
        kwargs['alpha'] = self.config.alpha
        kwargs['beta'] = self.config.beta

        # ! Temp Fix
        if model.__name__ == Custom.__name__:
            print('Custom Model Builder')
            arch = list(self.config.arch)
            self.config.num_layers = arch.count('g') + arch.count('p')

        # init model and optimizer and loss
        self.model = model(
            dataset.num_features,
            config.hidden_dim,
            dataset.num_classes,
            activation,
            layer=layer,
            arch=config.arch,
            num_layers=config.num_layers,
            dropout=config.dropout,
            input_norm=config.input_norm,
            layer_norm=config.layer_norm,
            *args,
            **kwargs,
        )

        ###########################################

        # Selecting Loss Function
        if self.config.loss == 'xentropy':
            loss_func = nn.CrossEntropyLoss
        elif self.config.loss == 'bceloss':
            loss_func = nn.BCEWithLogitsLoss
        else:
            raise NotImplementedError

        if self.config.optim == 'adam':
            optimizer = optim.Adam
        elif self.config.optim == 'sgd':
            optimizer = optim.SGD

        self.optimizer = optimizer(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.wd
        )

        self.loss_func = loss_func()

        # Set device
        if self.config.gpu >= 0:
            self.device = self.config.gpu
        else:
            self.device = 'cpu'

        # Move model
        self.model = self.model.to(self.device)

        # prepare trange
        self.tbar = trange(self.config.num_epochs, desc='Training Epochs')

        # create an empty stats logger
        self.stats = Stats(copy.deepcopy(self.config))

    def save(self, *args, **kwargs):
        self.stats.save()

    def calc_f1(self, pred, batch_labels):
        # import pdb; pdb.set_trace()
        
        if self.dataset.name.startswith('ogbn'):
            # dirty hack for now
            evaluator = Evaluator(name=self.dataset.name)
            if self.dataset.name == 'ogbn-proteins':
                pred_labels = pred.detach().cpu()
                score = evaluator.eval({
                    'y_true': batch_labels.cpu(),
                    'y_pred': pred_labels,
                })['rocauc']
            else:
                pred_labels = pred.detach().cpu().argmax(dim=1)
                score = evaluator.eval({
                    'y_true': batch_labels.cpu().reshape(-1, 1),
                    'y_pred': pred_labels.reshape(-1, 1),
                })['acc']
        else:
            # TODO: Fix multi-class score function
            pred_labels = pred.detach().cpu().argmax(dim=1)
            score = f1_score(batch_labels.cpu(), pred_labels, average="micro")

        return score

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def train(self, epoch):
        raise NotImplementedError

    def validation(self, epoch):
        raise NotImplementedError

    def inference(self):
        raise NotImplementedError

    def post_epoch(self, epoch):
        raise NotImplementedError

    def fetch_hiddens(self, ):
        raise NotImplementedError
