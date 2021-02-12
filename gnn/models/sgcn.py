
import torch
import torch.nn as nn

import torch.nn.functional as f

class SGCN(nn.Module):
    """
    https://arxiv.org/abs/1902.07153
    """

    def __init__(self,
                 features_dim,
                 hidden_dim,
                 num_classes,
                 activation,
                 layer=None,
                 arch='',
                 num_layers=0,
                 dropout=0,
                 *args,
                 **kwargs):

        super().__init__()

        self.num_layers = num_layers
        self.activation = activation
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        self.output_fc = nn.Linear(features_dim, num_classes)

        self.batch_norm = nn.BatchNorm1d(features_dim, affine=False)

        # TODO: Change this later to read arch and create corresponding layers
        for i in range(num_layers):
            self.layers.append(layer(features_dim, features_dim, layer_id=i+1))

    def forward(self, nodeblocks, x):

        # normalize input features
        h = self.batch_norm(x)
        h = self.dropout(h)
        for i, layer in enumerate(self.layers):
            h = layer(nodeblocks[i], h)
        h = self.output_fc(h)
        return h
