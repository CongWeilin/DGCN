
import os
import torch

from torch_geometric.datasets import *
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset

# https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html
def load(dataset_name, split=None):

    default_dir = os.path.join(os.path.expanduser('~'), '.gnn')
    dataset_dir = os.environ.get('GNN_DATASET_DIR', default_dir)

    # transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
    transform = T.Compose([T.ToSparseTensor()])

    if dataset_name in ['cora', 'citeseer', 'pubmed']: # public or full
        dataset = Planetoid(root=dataset_dir, name=dataset_name, split='full', pre_transform=transform)
    elif dataset_name in ['computers', 'photo']:
        dataset = Amazon(root=dataset_dir, name=dataset_name, pre_transform=transform)
    elif dataset_name == 'reddit':
        dataset = Reddit(root=dataset_dir+'/reddit/', pre_transform=transform)
        setattr(dataset, 'name', 'reddit')
    elif dataset_name.startswith('ogbn'):
        dataset = PygNodePropPredDataset(dataset_name, dataset_dir, pre_transform=transform)
        splitted_idx = dataset.get_idx_split()
        data = dataset.data

        # Fix few things about ogbn-proteins
        if dataset_name == 'ogbn-proteins':
            data.x = data.adj_t[0].mean(dim=1)
            data.adj_t[0].set_value_(None)
            dataset.slices['x'] = dataset.slices['y']
            dataset.__num_classes__ = 112 #WTF?
            data.y = data.y.to(torch.float)

        for split in ['train', 'valid', 'test']:
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[splitted_idx[split]] = True
            data[f'{split}_mask'] = mask
            dataset.slices[f'{split}_mask'] = dataset.slices['x']
        split = 'val'
        mask = data['valid_mask']
        data[f'{split}_mask'] = mask
        dataset.slices[f'{split}_mask'] = dataset.slices['x']        
        
    else:
        print('dataset {} is not supported!'.format(dataset_name))
        raise NotImplementedError

    return dataset

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def random_splits(data, num_classes, percls_train_num, valid_num):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    indices = []
    for i in range(num_classes):
        index = torch.nonzero((data.y == i)).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_train_num] for i in indices], dim=0)

    rest_index = torch.cat([i[percls_train_num:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:valid_num], size=data.num_nodes)
    data.test_mask = index_to_mask(
        rest_index[valid_num:], size=data.num_nodes)
    return data