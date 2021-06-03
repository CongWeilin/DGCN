import torch
import torch.nn as nn

import math

#########################################################
#########################################################
#########################################################
class GCN(nn.Module):
    def __init__(self, n_feat, n_hid, n_classes, n_layers, dropout, criterion):
        from layers import GraphConv
        super(GCN, self).__init__()
        self.n_layers = n_layers
        self.n_hid = n_hid
        
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConv(n_feat,  n_hid))
        for _ in range(n_layers-1):
            self.gcs.append(GraphConv(n_hid,  n_hid))
        self.linear = nn.Linear(n_hid, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.criterion = criterion

    def forward(self, x, adj):
        for ell in range(len(self.gcs)):
            x = self.gcs[ell](x, adj)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.linear(x)
        return x
    
    @torch.no_grad()
    def fetch_hiddens(self, x, adj):
        hiddens = []
        hiddens.append(x.clone().cpu().numpy())
        for ell in range(len(self.gcs)):
            x = self.gcs[ell](x, adj)
            hiddens.append(x.clone().cpu().numpy())
            x = self.relu(x)
            x = self.dropout(x)
        return hiddens
    
#########################################################
#########################################################
#########################################################
class GCNBias(nn.Module):
    def __init__(self, n_feat, n_hid, n_classes, n_layers, dropout, criterion):
        from layers import GraphConv
        super(GCNBias, self).__init__()
        self.n_layers = n_layers
        self.n_hid = n_hid
        
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConv(n_feat,  n_hid, bias=True))
        for _ in range(n_layers-1):
            self.gcs.append(GraphConv(n_hid,  n_hid, bias=True))
        self.linear = nn.Linear(n_hid, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.criterion = criterion

    def forward(self, x, adj):
        for ell in range(len(self.gcs)):
            x = self.gcs[ell](x, adj)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.linear(x)
        return x
    
    @torch.no_grad()
    def fetch_hiddens(self, x, adj):
        hiddens = []
        hiddens.append(x.clone().cpu().numpy())
        for ell in range(len(self.gcs)):
            x = self.gcs[ell](x, adj)
            hiddens.append(x.clone().cpu().numpy())
            x = self.relu(x)
            x = self.dropout(x)
        return hiddens

#########################################################
#########################################################
#########################################################    

class SGC(nn.Module):
    def __init__(self, n_feat, n_hid, n_classes, n_layers, dropout, criterion):
        super(SGC, self).__init__()
        from layers import SGCLayer
        self.n_layers = n_layers
        self.n_hid = n_hid
        
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            self.gcs.append(SGCLayer())
        self.linear_in = nn.Linear(n_feat, n_hid)
        self.linear_out = nn.Linear(n_hid, n_classes)
        # self.linear_out = nn.Linear(n_feat, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.criterion = criterion

    def forward(self, x, adj):
        x = self.linear_in(x)
        for ell in range(len(self.gcs)):
            x = self.gcs[ell](x, adj)
        x = self.linear_out(x)
        return x
    
    @torch.no_grad()
    def fetch_hiddens(self, x, adj):
        hiddens = []
        hiddens.append(x.clone().cpu().numpy())
        x = self.linear_in(x)
        for ell in range(len(self.gcs)):
            x = self.gcs[ell](x, adj)
            hiddens.append(x.clone().cpu().numpy())
        x = self.linear_out(x)
        return hiddens
#########################################################
#########################################################
#########################################################    

class ResGCN(nn.Module):
    def __init__(self, n_feat, n_hid, n_classes, n_layers, dropout, criterion):
        from layers import GraphConv
        super(ResGCN, self).__init__()
        self.n_layers = n_layers
        self.n_hid = n_hid
        
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConv(n_feat,  n_hid))
        for _ in range(n_layers-1):
            self.gcs.append(GraphConv(n_hid,  n_hid))
        self.linear = nn.Linear(n_hid, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.criterion = criterion

    def forward(self, x, adj):
        for ell in range(len(self.gcs)):
            if ell > 0:
                x_res = x.clone()
                x = self.gcs[ell](x, adj)
                x = self.relu(x)
                x = self.dropout(x) + x_res
            else:
                x = self.gcs[ell](x, adj)
                x = self.relu(x)
                x = self.dropout(x)
        x = self.linear(x)
        return x
    
    @torch.no_grad()
    def fetch_hiddens(self, x, adj):
        hiddens = []
        hiddens.append(x.clone().cpu().numpy())
        for ell in range(len(self.gcs)):
            if ell > 0:
                x_res = x.clone()
                x = self.gcs[ell](x, adj)
                hiddens.append(x.clone().cpu().numpy())
                x = self.relu(x)
                x = self.dropout(x) + x_res
            else:
                x = self.gcs[ell](x, adj)
                hiddens.append(x.clone().cpu().numpy())
                x = self.relu(x)
                x = self.dropout(x)
        return hiddens
    
#########################################################
#########################################################
#########################################################    

class GCNII(nn.Module):
    def __init__(self, n_feat, n_hid, n_classes, n_layers, dropout, criterion):
        from layers import GCNIILayer
        super(GCNII, self).__init__()
        self.n_layers = n_layers
        self.n_hid = n_hid
        
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            self.gcs.append(GCNIILayer(n_hid,  n_hid))
        self.linear_in = nn.Linear(n_feat, n_hid)
        self.linear_out = nn.Linear(n_hid, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.criterion = criterion

    def get_alpha_beta(self, ell):
        alpha = 0.8
        beta = math.log(0.5/(ell+1)+1)
        return alpha, beta
    
    def forward(self, x, adj):
        x = self.linear_in(x)
        x_0 = x.clone()
        for ell in range(len(self.gcs)):
            alpha, beta = self.get_alpha_beta(ell)
            x = self.gcs[ell](x, adj, x_0, alpha, beta)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.linear_out(x)
        return x
    
    @torch.no_grad()
    def fetch_hiddens(self, x, adj):
        hiddens = []
        hiddens.append(x.clone().cpu().numpy())
        x = self.linear_in(x)
        x_0 = x.clone()
        for ell in range(len(self.gcs)):
            alpha, beta = self.get_alpha_beta(ell)
            x = self.gcs[ell](x, adj, x_0, alpha, beta)
            hiddens.append(x.clone().cpu().numpy())
            x = self.relu(x)
            x = self.dropout(x)
        return hiddens
            
#########################################################
#########################################################
#########################################################    

class APPNP(nn.Module):
    def __init__(self, n_feat, n_hid, n_classes, n_layers, dropout, criterion):
        from layers import APPNPLayer
        super(APPNP, self).__init__()
        self.n_layers = n_layers
        self.n_hid = n_hid
        
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            self.gcs.append(APPNPLayer())
        self.linear_in = nn.Linear(n_feat, n_hid)
        self.linear_out = nn.Linear(n_hid, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.criterion = criterion

    def forward(self, x, adj):
        x = self.linear_in(x)
        x_0 = x.clone()
        for ell in range(len(self.gcs)):
            alpha = 0.8
            x = self.gcs[ell](x, adj, x_0, alpha)
        x = self.linear_out(x)
        return x
    
    @torch.no_grad()
    def fetch_hiddens(self, x, adj):
        hiddens = []
        hiddens.append(x.clone().cpu().numpy())
        x = self.linear_in(x)
        x_0 = x.clone()
        for ell in range(len(self.gcs)):
            alpha = 0.8
            x = self.gcs[ell](x, adj, x_0, alpha)
            hiddens.append(x.clone().cpu().numpy())
        return hiddens
    
    
#########################################################
#########################################################
#########################################################
class MLP(nn.Module):
    def __init__(self, n_feat, n_hid, n_classes, n_layers, dropout, criterion):
        super(MLP, self).__init__()
        self.n_layers = n_layers
        self.n_hid = n_hid
        
        self.gcs = nn.ModuleList()
        self.gcs.append(nn.Linear(n_feat,  n_hid))
        for _ in range(n_layers-1):
            self.gcs.append(nn.Linear(n_hid,  n_hid))
        self.linear = nn.Linear(n_hid, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.criterion = criterion

    def forward(self, x, adj):
        for ell in range(len(self.gcs)):
            x = self.gcs[ell](x)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.linear(x)
        return x
    
    @torch.no_grad()
    def fetch_hiddens(self, x, adj):
        hiddens = []
        hiddens.append(x.clone().cpu().numpy())
        for ell in range(len(self.gcs)):
            x = self.gcs[ell](x)
            hiddens.append(x.clone().cpu().numpy())
            x = self.relu(x)
            x = self.dropout(x)
        return hiddens