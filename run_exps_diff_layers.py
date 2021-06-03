import torch
import torch.nn as nn
import torch.optim as optim

from utils import load_data, process_graph_data
from utils import package_mxl, adj_rw_norm
from utils import sparse_mx_to_torch_sparse_tensor
from utils import ResultRecorder

from model import GCN, GCNBias, SGC, ResGCN, GCNII, APPNP
from layers import GraphConv
from load_semigcn_data import load_data_gcn
from data_loader import DataLoader

from sklearn.metrics import f1_score
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.sparse.csgraph import connected_components

from tqdm import trange
import numpy as np
import copy 
import time
import pickle
import os


import argparse
"""
Dataset arguments
"""
parser = argparse.ArgumentParser(
    description='Training GCN on Large-scale Graph Datasets')
parser.add_argument('--dataset', type=str, default='cora',
                    help='Dataset name: pubmed/flickr/reddit/ppi-large')
parser.add_argument('--method', type=str, default='ResGCN/GCN/SGC',
                    help='Algorithms: seperate using slash')
parser.add_argument('--nhid', type=int, default=64,
                    help='Hidden state dimension')
parser.add_argument('--epoch_num', type=int, default=500,
                    help='Number of Epoch')
parser.add_argument('--batch_size', type=int, default=99999,
                    help='size of output node in a batch')
parser.add_argument('--n_layers', type=int, default=5,
                    help='Number of GCN layers')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
args = parser.parse_args()
print(args)

method = args.method.split('/')

"""
Prepare devices
"""
if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")
    
if args.dataset not in ['cora', 'citeseer', 'pubmed']:
    temp_data = load_data(args.dataset) 
else:
    temp_data = load_data_gcn(args.dataset)

adj_full, adj_train, feat_data, labels, role = process_graph_data(*temp_data)

train_nodes = np.array(role['tr'])
valid_nodes = np.array(role['va'])
test_nodes = np.array(role['te'])

data_loader = DataLoader(adj_full, train_nodes, valid_nodes, test_nodes, device)

def get_proj_mat(data_loader):
    adj_mat = data_loader.adj_mat.toarray()
    num_nodes = data_loader.num_nodes
    n_components, components_label = connected_components(adj_mat)
    E_mat = np.zeros((n_components, num_nodes))
    for node_i in range(num_nodes):
        deg = adj_mat[node_i, :].sum()
        E_mat[components_label[node_i], node_i] = 1/np.sqrt(deg)
    E_mat = (E_mat.T/E_mat.sum(axis=1)).T

    P_mat = np.matmul(E_mat.T, np.linalg.inv(E_mat.dot(E_mat.T))).dot(E_mat)
    F_mat = np.eye(P_mat.shape[0]) - P_mat
    return F_mat

def compute_dM(model, F_mat, feat_data_th, data_loader):
    hiddens = model.fetch_hiddens(feat_data_th, data_loader.lap_tensor)
    dM = [np.linalg.norm(F_mat.dot(hidden)) for hidden in hiddens]
    # dM = np.array(dM)/dM[0]
    return dM

def get_weight_norms(model):
    weight_norms = dict()
    for n, p in model.named_parameters():
        if 'weight' in n and 'gcs' in n:
            weight_norms[n] = p.data.norm(2).item()
    return weight_norms

def get_weight_sigval(model):
    weight_sigval= dict()
    for n, p in model.named_parameters():
        if 'weight' in n and 'gcs' in n:
            U, S, V = torch.svd(p.data, compute_uv=False)
            weight_sigval[n] = S.max().item()
    return weight_sigval

def get_grad_norms(model):
    grad_norms = dict()
    for n, p in model.named_parameters():
        grad_norms[n] = p.grad.data.norm(2).item()
    return grad_norms
        
def compute_hidden_dist(model, feat_data_th, data_loader):
    unique_labels = np.arange(num_classes)
    class_to_indices = dict()
    for i in unique_labels:
        class_to_indices[i] = np.where(np.argmax(labels, axis=1)==i)[0]

    inner_class_dist_list = []
    cross_class_dist_list = []
    for hiddens in model.fetch_hiddens(feat_data_th, data_loader.lap_tensor):
        hiddens_norm = np.linalg.norm(hiddens)

        inner_class_dist = []
        for i in unique_labels:
            indices = class_to_indices[i]
            dists = pairwise_distances(hiddens[indices])
            inner_class_dist.append(dists.mean())
        inner_class_dist_list += [np.mean(inner_class_dist)/hiddens_norm]

        cross_class_dist = []
        all_nodes_indices = np.arange(len(labels))
        for i in unique_labels:
            indices = class_to_indices[i]
            other_indices = np.setdiff1d(all_nodes_indices, indices)
            dists = pairwise_distances(hiddens[indices], hiddens[other_indices])
            cross_class_dist.append(dists.mean())
        cross_class_dist_list += [np.mean(cross_class_dist)/hiddens_norm]
    return inner_class_dist_list, cross_class_dist_list

"""
Setup datasets and models for training (multi-class use sigmoid+binary_cross_entropy, use softmax+nll_loss otherwise)
"""

if args.dataset in ['flickr', 'reddit', 'cora', 'citeseer', 'pubmed']:
    feat_data_th = torch.FloatTensor(feat_data)
    labels_th = torch.LongTensor(labels.argmax(1))
    num_classes = labels_th.max().item()+1
    criterion = nn.CrossEntropyLoss()
    multi_class=False
elif args.dataset in ['ppi', 'ppi-large', 'amazon', 'yelp']:
    feat_data_th = torch.FloatTensor(feat_data)
    labels_th = torch.FloatTensor(labels)
    num_classes = labels_th.shape[1]
    criterion = nn.BCEWithLogitsLoss()
    multi_class=True

feat_data_th = feat_data_th.to(device)
labels_th = labels_th.to(device)

def sgd_step(net, optimizer, feat_data, labels, train_data, device):
    """
    Function to updated weights with a SGD backpropagation
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    net.train()
    epoch_loss = []
    epoch_acc = []
    
    # Run over the train_loader
    mini_batches, adj = train_data
    for mini_batch in mini_batches:

        # compute current stochastic gradient
        optimizer.zero_grad()
        output = net(feat_data, adj)
        
        loss = net.criterion(output[mini_batch], labels[mini_batch])
        loss.backward()
        
        optimizer.step()
        epoch_loss.append(loss.item())
        
        if multi_class:
            output[output > 0.5] = 1
            output[output <= 0.5] = 0
        else:
            output = output.argmax(dim=1)

        acc = f1_score(output[mini_batch].detach().cpu(), 
                       labels[mini_batch].detach().cpu(), average="micro")
        epoch_acc.append(acc)

    return epoch_loss, epoch_acc

@torch.no_grad()
def inference(eval_model, feat_data, labels, test_data, device):
    eval_model = eval_model.to(device)
    mini_batch, adj = test_data    
    output = eval_model(feat_data, adj)
    loss = eval_model.criterion(output[mini_batch], labels[mini_batch]).item()
    
    if multi_class:
        output[output > 0.5] = 1
        output[output <= 0.5] = 0
    else:
        output = output.argmax(dim=1)
        
    acc = f1_score(output[mini_batch].detach().cpu(), 
                   labels[mini_batch].detach().cpu(), average="micro")
    return loss, acc

"""
Train without sampling
"""

def train_model(model, data_loader, note):
    train_model = copy.deepcopy(model).to(device)
    
    results = ResultRecorder(note=note)

    optimizer = optim.Adam(train_model.parameters())

    tbar = trange(args.epoch_num, desc='Training Epochs')
    for epoch in tbar:
        # fetch train data 
        
        sample_time_st = time.perf_counter()
        train_data = data_loader.get_mini_batches(batch_size=args.batch_size)
        sample_time = time.perf_counter() - sample_time_st
        
        compute_time_st = time.perf_counter()
        train_loss, train_acc = sgd_step(train_model, optimizer, feat_data_th, labels_th, train_data, device)
        compute_time = time.perf_counter() - compute_time_st
        
        epoch_train_loss = np.mean(train_loss)
        epoch_train_acc = np.mean(train_acc)

        valid_data = data_loader.get_valid_batch()
        epoch_valid_loss, epoch_valid_acc = inference(train_model, feat_data_th, labels_th, valid_data, device)
        tbar.set_postfix(acc=train_acc,
                         loss=epoch_train_loss,
                         val_loss=epoch_valid_loss,
                         val_score=epoch_valid_acc)

        results.update(epoch_train_loss, 
                       epoch_train_acc,
                       epoch_valid_loss, 
                       epoch_valid_acc, 
                       train_model, sample_time=sample_time, compute_time=compute_time)


    test_data = data_loader.get_test_batch()
    epoch_test_loss, epoch_test_acc = inference(results.best_model, feat_data_th, labels_th, test_data, device)
    results.test_loss = epoch_test_loss
    results.test_acc = epoch_test_acc
    print('Test_loss: %.4f | test_acc: %.4f' % (epoch_test_loss, epoch_test_acc))
    
    print('Average sampling time %.5fs, average computing time %.5fs'%
          (np.mean(results.sample_time), np.mean(results.compute_time)))
    
    return results

F_mat = get_proj_mat(data_loader)
for repeat in range(10):
    results_list = []
    if 'SGC' in method:
        model = SGC(n_feat=feat_data.shape[1], 
                    n_hid=args.nhid, 
                    n_classes=num_classes, 
                    n_layers=args.n_layers, 
                    dropout=args.dropout, 
                    criterion=criterion)
        model_untrain = copy.deepcopy(model)
        results = train_model(model, data_loader, note="SGC (L=%d)"%args.n_layers)
        results.dM_before = compute_dM(model_untrain.to(device), F_mat, feat_data_th, data_loader)
        results.dM_after = compute_dM(results.best_model.to(device), F_mat, feat_data_th, data_loader)
        results.w_sigval_before = get_weight_sigval(model_untrain)
        results.w_sigval_after = get_weight_sigval(results.best_model)
        results.inner_dist, results.cross_dist = compute_hidden_dist(model_untrain.to(device), feat_data_th, data_loader)
        results.inner_dist_after, results.cross_dist_after = compute_hidden_dist(results.best_model.to(device), feat_data_th, data_loader)
        results_list.append(results)
    if 'GCN' in method:
        model = GCN(n_feat=feat_data.shape[1], 
                    n_hid=args.nhid, 
                    n_classes=num_classes, 
                    n_layers=args.n_layers, 
                    dropout=args.dropout, 
                    criterion=criterion)
        model_untrain = copy.deepcopy(model)
        results = train_model(model, data_loader, note="GCN (L=%d)"%args.n_layers)
        results.dM_before = compute_dM(model_untrain.to(device), F_mat, feat_data_th, data_loader)
        results.dM_after = compute_dM(results.best_model.to(device), F_mat, feat_data_th, data_loader)
        results.w_sigval_before = get_weight_sigval(model_untrain)
        results.w_sigval_after = get_weight_sigval(results.best_model)
        results.inner_dist, results.cross_dist = compute_hidden_dist(model_untrain.to(device), feat_data_th, data_loader)
        results.inner_dist_after, results.cross_dist_after = compute_hidden_dist(results.best_model.to(device), feat_data_th, data_loader)
        results_list.append(results)
    if 'GCNBias' in method:
        model = GCNBias(n_feat=feat_data.shape[1], 
                    n_hid=args.nhid, 
                    n_classes=num_classes, 
                    n_layers=args.n_layers, 
                    dropout=args.dropout, 
                    criterion=criterion)
        model_untrain = copy.deepcopy(model)
        results = train_model(model, data_loader, note="GCNBias (L=%d)"%args.n_layers)
        results.dM_before = compute_dM(model_untrain.to(device), F_mat, feat_data_th, data_loader)
        results.dM_after = compute_dM(results.best_model.to(device), F_mat, feat_data_th, data_loader)
        results.w_sigval_before = get_weight_sigval(model_untrain)
        results.w_sigval_after = get_weight_sigval(results.best_model)
        results.inner_dist, results.cross_dist = compute_hidden_dist(model_untrain.to(device), feat_data_th, data_loader)
        results.inner_dist_after, results.cross_dist_after = compute_hidden_dist(results.best_model.to(device), feat_data_th, data_loader)
        results_list.append(results)
    if 'ResGCN' in method:
        model = ResGCN(n_feat=feat_data.shape[1], 
                    n_hid=args.nhid, 
                    n_classes=num_classes, 
                    n_layers=args.n_layers, 
                    dropout=args.dropout, 
                    criterion=criterion)
        model_untrain = copy.deepcopy(model)
        results = train_model(model, data_loader, note="ResGCN (L=%d)"%args.n_layers)
        results.dM_before = compute_dM(model_untrain.to(device), F_mat, feat_data_th, data_loader)
        results.dM_after = compute_dM(results.best_model.to(device), F_mat, feat_data_th, data_loader)
        results.w_sigval_before = get_weight_sigval(model_untrain)
        results.w_sigval_after = get_weight_sigval(results.best_model)
        results.inner_dist, results.cross_dist = compute_hidden_dist(model_untrain.to(device), feat_data_th, data_loader)
        results.inner_dist_after, results.cross_dist_after = compute_hidden_dist(results.best_model.to(device), feat_data_th, data_loader)
        results_list.append(results)
    if 'APPNP' in method:
        model = APPNP(n_feat=feat_data.shape[1], 
                    n_hid=args.nhid, 
                    n_classes=num_classes, 
                    n_layers=args.n_layers, 
                    dropout=args.dropout, 
                    criterion=criterion)
        model_untrain = copy.deepcopy(model)
        results = train_model(model, data_loader, note="APPNP (L=%d)"%args.n_layers)
        results.dM_before = compute_dM(model_untrain.to(device), F_mat, feat_data_th, data_loader)
        results.dM_after = compute_dM(results.best_model.to(device), F_mat, feat_data_th, data_loader)
        results.w_sigval_before = get_weight_sigval(model_untrain)
        results.w_sigval_after = get_weight_sigval(results.best_model)
        results.inner_dist, results.cross_dist = compute_hidden_dist(model_untrain.to(device), feat_data_th, data_loader)
        results.inner_dist_after, results.cross_dist_after = compute_hidden_dist(results.best_model.to(device), feat_data_th, data_loader)
        results_list.append(results)
    if 'GCNII' in method:
        model = GCNII(n_feat=feat_data.shape[1], 
                    n_hid=args.nhid, 
                    n_classes=num_classes, 
                    n_layers=args.n_layers, 
                    dropout=args.dropout, 
                    criterion=criterion)
        model_untrain = copy.deepcopy(model)
        results = train_model(model, data_loader, note="GCNII (L=%d)"%args.n_layers)
        results.dM_before = compute_dM(model_untrain.to(device), F_mat, feat_data_th, data_loader)
        results.dM_after = compute_dM(results.best_model.to(device), F_mat, feat_data_th, data_loader)
        results.w_sigval_before = get_weight_sigval(model_untrain)
        results.w_sigval_after = get_weight_sigval(results.best_model)
        results.inner_dist, results.cross_dist = compute_hidden_dist(model_untrain.to(device), feat_data_th, data_loader)
        results.inner_dist_after, results.cross_dist_after = compute_hidden_dist(results.best_model.to(device), feat_data_th, data_loader)
        results_list.append(results)
    
    save_path = os.path.join('./exp_results/%s/'%args.dataset, 
                             'results_%s_L%d_repteat%d.pkl'%(args.dataset, args.n_layers, repeat))
    
    with open(save_path, 'wb') as f:
        pickle.dump(results_list, f)