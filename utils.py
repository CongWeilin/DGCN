import numpy as np
import json
import copy
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Load data
"""
def load_data(prefix, normalize=True):
    adj_full = sp.load_npz('./data/{}/adj_full.npz'.format(prefix)).astype(np.bool)
    adj_train = sp.load_npz('./data/{}/adj_train.npz'.format(prefix)).astype(np.bool)
    role = json.load(open('./data/{}/role.json'.format(prefix)))
    feats = np.load('./data/{}/feats.npy'.format(prefix))
    class_map = json.load(open('./data/{}/class_map.json'.format(prefix)))
    class_map = {int(k):v for k,v in class_map.items()}
    assert len(class_map) == feats.shape[0]
    # ---- normalize feats ----
    train_nodes = np.array(list(set(adj_train.nonzero()[0])))
    train_feats = feats[train_nodes]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)
    # -------------------------
    return adj_full, adj_train, feats, class_map, role

def process_graph_data(adj_full, adj_train, feats, class_map, role):
    num_vertices = adj_full.shape[0]
    if isinstance(list(class_map.values())[0],list):
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_vertices, num_classes))
        for k,v in class_map.items():
            class_arr[k] = v
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = np.zeros((num_vertices, num_classes))
        offset = min(class_map.values())
        for k,v in class_map.items():
            class_arr[k][v-offset] = 1
    return adj_full, adj_train, feats, class_arr, role

"""
Prepare adjacency matrices
"""
def package_mxl(mxl, device):
    return [torch.sparse.FloatTensor(mx[0], mx[1], mx[2]).to(device) for mx in mxl]

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 and len(sparse_mx.col) == 0:
        indices = torch.LongTensor([[], []])
    else:
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return indices, values, shape

def adj_rw_norm(adj):
    """
    Normalize adj according to the method of rw normalization.
    """
    diag_shape = (adj.shape[0],adj.shape[1])
    D = adj.sum(1).flatten()
    norm_diag = sp.dia_matrix((1/D,0),shape=diag_shape)
    adj_norm = norm_diag.dot(adj)
    return adj_norm


"""
Record the best model
"""
class ResultRecorder:
    def __init__(self, note):
        self.train_loss_record = []
        self.train_acc_record = []
        self.loss_record = []
        self.acc_record = []
        self.best_acc = None
        self.best_model = None
        self.note = note
        self.sample_time = []
        self.compute_time = []
        self.state_dicts = []
        self.grad_norms = []
        
    def update(self, train_loss, train_acc, loss, acc, model, sample_time=0, compute_time=0):
        self.sample_time += [sample_time]
        self.compute_time += [compute_time]

        self.train_loss_record += [train_loss]
        self.train_acc_record += [train_acc]
        
        self.loss_record += [loss]
        self.acc_record += [acc]
            
        if self.best_acc is None:
            self.best_acc = acc
        elif self.best_acc < acc:
            self.best_acc = acc
            self.best_model = copy.deepcopy(model).cpu()
            