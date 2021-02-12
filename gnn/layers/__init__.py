from .gconv import GConv
from .gconvii import GConvII
from .apnpp_conv import APPNP_Conv
from .sgconv import SGConv
from .gprop import GProp
from .pairnorm import PairNorm
from .local_gconv import LocalGConv

def layer_selector(layer_str):

    if layer_str == 'gconv':
        layer = GConv
    elif layer_str == 'gconvii':
        layer = GConvII
    elif layer_str == 'appnp_conv':
        layer = APPNP_Conv
    elif layer_str == 'sgconv':
        layer = SGConv
    elif layer_str == 'gprop':
        layer = GProp
    elif layer_str == 'local_gconv':
        layer = LocalGConv
    else:
        return NotImplementedError
    
    return layer