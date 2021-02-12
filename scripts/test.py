import os
import sys

sys.path.append('..')
from gnn import datasets, layers, models, training, utils

from helpers import run_repeated

if os.environ['LOGNAME'] == 'mfr5226':
    os.environ['GNN_DATASET_DIR'] = '/export/local/mfr5226/datasets/pyg/'
else:
    os.environ['GNN_DATASET_DIR'] = '/home/weilin/Downloads/GCN_datasets/'


dataset = 'cora'
out = sys.argv[1]
num_runs = int(sys.argv[2])

output_dir = '/export/local/mfr5226/outputs/next/{}/'.format(out)

# load the graph
dataset = datasets.load(dataset)

config_dict = {
    'loss': 'xentropy',
    'optim': 'adam',
    'activation': 'relu',
    'gpu': 0,
    'num_epochs': 500,
    'hidden_dim': 128,
    'lr': 1e-3,
    'dropout': 0.1,
    'wd': 0,
    'dataset': dataset,
    'adj_norm': 'sym',
    'output_dir': output_dir,
}


# GCN, 72?
new_config = {
    'model': 'gcn',
    'layer': 'gconv',
    'num_epochs': 10,
    'dropout': 0.0,
    'val_steps': 5,
    'lr': 1e-2,
    'num_layers': 3,
    'batch_norm': False,
    'run_name': 'gcn'
}

config_dict.update(new_config)
run_repeated(num_runs, config_dict, dataset)