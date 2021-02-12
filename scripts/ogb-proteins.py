import os
import sys
from helpers import run_repeated

sys.path.append('..')
from gnn import datasets, layers, models, training, utils

if os.environ['LOGNAME'] == 'mfr5226':
    os.environ['GNN_DATASET_DIR'] = '/export/local/mfr5226/datasets/pyg/'
else:
    os.environ['GNN_DATASET_DIR'] = '/home/weilin/Downloads/GCN_datasets/'


# read input arguments using utils
dataset = 'ogbn-proteins'
out = sys.argv[1]
num_runs = int(sys.argv[2])

output_dir = '/export/local/mfr5226/outputs/next/{}/'.format(out)

# load the graph
dataset = datasets.load(dataset)

config_dict = {
    'loss': 'bceloss',
    'optim': 'adam',
    'activation': 'relu',
    'gpu': 0,
    'num_epochs': 1000,
    'hidden_dim': 128,
    'lr': 1e-2,
    'dropout': 0,
    'wd': 0,
    'val_steps': 5,
    'batch_norm': False,
    'dataset': dataset,
    'adj_norm': 'sym',
    'output_dir': output_dir,
}


# GCN, 
# new_config = {
#     'model': 'gcn',
#     'layer': 'gconv',
#     'num_layers': 3,
#     'run_name': 'gcn'
# }
# config_dict.update(new_config)
# run_repeated(num_runs, config_dict, dataset)

# ResGCN, 
# new_config = {
#     'model': 'resgcn',
#     'layer': 'gconv',
#     'num_layers': 6,
#     'run_name': 'resgcn'
# }
# config_dict.update(new_config)
# run_repeated(num_runs, config_dict, dataset)


# GCNII, 
# new_config = {
#     'model': 'gcnii',
#     'layer': 'gconvii',
#     'num_layers': 6,
#     'alpha': 0.5,
#     'dropout': 0.1,
#     'run_name': 'gcnii'
# }
# config_dict.update(new_config)
# run_repeated(num_runs, config_dict, dataset)


# # APPNP,
# new_config = {
#     'model': 'appnp',
#     'layer': 'appnp_conv',
#     'num_layers': 8,
#     'alpha': 0.2,
#     'run_name': 'appnp'
# }
# config_dict.update(new_config)
# run_repeated(num_runs, config_dict, dataset)


# # DGCN
new_config = {
    'model': 'local_gcn_ogb_arxiv',
    'layer': 'local_gconv',
    'num_layers': 6,
    'alpha': 0.2,
    'dropout': 0.1,
    'run_name': 'dgcn'
}
config_dict.update(new_config)
run_repeated(num_runs, config_dict, dataset)

##### test runs ####
# config_dict.update(new_config)
# config = utils.Config(config_dict)
# trainer = training.Full(config, dataset)
# print(trainer.model)
# # import pdb; pdb.set_trace()
# trainer.run()
# trainer.save()
