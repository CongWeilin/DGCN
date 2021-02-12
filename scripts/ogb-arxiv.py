import os
import sys
from helpers import run_repeated

sys.path.append('..')
from gnn import datasets, layers, models, training, utils

# read input arguments using utils
dataset = 'ogbn-arxiv'
out = sys.argv[1]
num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
model = sys.argv[3] if len(sys.argv) > 3 else 'gcn'
num_layers = int(sys.argv[4]) if len(sys.argv) > 4 else 2

print(out, num_runs, model, num_layers)

if os.environ['LOGNAME'] == 'mfr5226':
    os.environ['GNN_DATASET_DIR'] = '/export/local/mfr5226/datasets/pyg/'
    output_dir = '/export/local/mfr5226/outputs/next/{}/'.format(out)
else:
    os.environ['GNN_DATASET_DIR'] = '/home/weilin/Downloads/GCN_datasets/'
    output_dir = out

# load the graph
dataset = datasets.load(dataset)

config_dict = {
    'loss': 'xentropy',
    'optim': 'adam',
    'activation': 'relu',
    'gpu': 0,
    'num_epochs': 500,
    'hidden_dim': 128,
    'alpha': 0.5,
    'dropout': 0.5,
    'val_steps': 1,
    'lr': 1e-2,
    'wd': 0,
    'input_norm': False,
    'layer_norm': True,
    'dataset': dataset,
    'adj_norm': 'sym',
    'output_dir': output_dir,
}

if model == 'gcn':
    # GCN
    new_config = {
        'model': 'gcn',
        'layer': 'gconv',
        'num_layers': num_layers,
        'run_name': 'gcn'
    }
elif model == 'resgcn':
    # RsGCN
    new_config = {
        'model': 'resgcn_ogb',
        'layer': 'gconv',
        'num_layers': num_layers,
        'run_name': 'resgcn'
    }
elif model == 'gcnii':
    # GCNII
    new_config = {
        'model': 'gcnii_ogb',
        'layer': 'gconvii',
        'num_layers': num_layers,
        'run_name': 'gcnii'
    }
elif model == 'appnp':
    # APPNP
    new_config = {
        'model': 'appnp_ogb',
        'layer': 'appnp_conv',
        'num_layers': num_layers,
        'run_name': 'appnp'
    }
elif model == 'dgcn':
    # DecoupledGCN
    new_config = {
        'model': 'decoupledgcn_ogb',
        'layer': 'local_gconv',
        'num_layers': num_layers,
        'run_name': 'dgcn'
    }

config_dict.update(new_config)
run_repeated(num_runs, config_dict, dataset)