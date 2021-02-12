
import torch
import torch.nn as nn


class Base(nn.Module):
    """Base class for all GNN models
    """

    def __init__(self):
        pass

    def build(self, arch_str, layer_str, prop_str):
        pass


def model_builder(arch, gconv, prop, feat_dim, hid_dim, out_dim, *args, **kwargs):

    layers_list = nn.ModuleList()
    layers_tokens = list(arch)
    num_layers = len(layers_tokens)

    input_dim = feat_dim
    output_dim = hid_dim if num_layers > 1 else out_dim
    layer = None
    
    for i, l in enumerate(layers_tokens):
        
        if l == 'l':
            layer = nn.Linear(input_dim, output_dim)
        elif l == 'g':
            layer = gconv(input_dim, output_dim, layer_id=i, *args, **kwargs)
        elif l == 'p':
            layer = prop(input_dim, output_dim, layer_id=i, *args, **kwargs)

        layers_list.append(layer)

        input_dim = hid_dim
        output_dim = hid_dim if i + 1 < num_layers - 1 else out_dim


    return layers_list