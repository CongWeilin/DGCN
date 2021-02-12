# from .base import Base
from .gcn import GCN, GCN_Paper, GCN_OGB
from .gcnii import GCNII, GCNII_OGB
from .appnp import APPNP_Paper, APPNP_OGB
from .res_gcn import ResGCN, ResGCN_OGB
from .decoupled_gcn import DecoupledGCN, DecoupledGCN_OGB
from .sgcn import SGCN
from .pairnorm import GCN_Pairnorm
from .custom import Custom


def model_selector(model_str):
     # Selecting Model
    if model_str == 'gcn':
        model = GCN
    elif model_str == 'gcn_paper':
        model = GCN_Paper
    elif model_str == 'gcn_ogb':
        model = GCN_OGB
    elif model_str == 'gcnii':
        model = GCNII
    elif model_str == 'gcnii_ogb':
        model = GCNII_OGB
    # elif model_str == 'gcnii_arxiv':
    #     model = GCNII_Arxiv
    elif model_str == 'appnp':
        model = APPNP_Paper
    elif model_str == 'appnp_ogb':
        model = APPNP_OGB
    elif model_str == 'resgcn':
        model = ResGCN
    elif model_str == 'resgcn_ogb':
        model = ResGCN_OGB
    elif model_str == 'sgcn':
        model = SGCN
    elif model_str == 'pairnorm':
        model = GCN_Pairnorm
    elif model_str == 'decoupledgcn':
        model = DecoupledGCN
    elif model_str == 'decoupledgcn_ogb':
        model = DecoupledGCN_OGB
    elif model_str == 'custom':
        model = Custom
    else:
        return NotImplementedError

    return model