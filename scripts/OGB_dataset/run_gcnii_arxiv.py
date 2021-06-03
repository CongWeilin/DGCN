import os
import sys
import argparse

sys.path.append('../../..')
os.environ['GNN_DATASET_DIR'] = '/home/weilin/Downloads/GCN_datasets/'

from gnn import datasets, layers, models, training, utils
import numpy as np

def print_stat(result_stat):
    best_results = []
    for r in result_stat:
        train = np.max(r.train_scores) 
        valid = np.max(r.val_scores) 
        test = r.test_score
        best_results.append((train, valid, test))

    best_result = np.array(best_results) * 100

    print(f'All runs:')

    r = best_result[:, 0]
    print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
    r = best_result[:, 1]
    print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
    r = best_result[:, 2]
    print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
    parser.add_argument('--num_layers', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--create_split', type=str, default='./')
    
    args = parser.parse_args()
    
    dataset = datasets.load(args.dataset)
    
    config_dict = {
        'model': 'gcnii_ogb_arxiv',
        'layer': 'gconvii',
        'loss': 'xentropy',
        'optim': 'adam',
        'activation': 'relu',
        'num_epochs': 500,
        'lr': 1e-2,
        'dropout': 0.5,
        'wd': 0,
        'adj_norm': 'sym',
        'gpu': args.gpu,
        'num_layers': args.num_layers,
        'hidden_dim': args.hidden_dim,
        'dataset': dataset,
        'output_dir': args.output_dir,
        'create_split': args.create_split,
        'alpha': 0.2,
        'beta': 0.5,
    }
    
    result_trainer = list()
    for repeat in range(10):
        config = utils.Config(config_dict)
        trainer = training.Full(config, dataset)
        print(trainer.model)
        trainer.run()
        result_trainer.append(trainer.stats)
        
    import pickle
    save_path = os.path.join(args.output_dir, 
                             'dataset_%s_layer_%d'%(args.dataset, args.num_layers))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, 'gcnii.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(result_trainer, f)
    print(save_path)
    print_stat(result_trainer)