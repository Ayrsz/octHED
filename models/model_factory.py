
from models.decoder_model import UncertHED
from models.exitationhed_model import EXITHED
from models.octave_model import OCTHED
from models.hed_model import HED
from models.fourier_model import FFCHED
from models.side_outputs_dense_model import DENSEHED
from models.decoder_model_octave import UNCERTHED_OCTAVE

import torch.nn as nn

def get_model(args, device = 'cpu'):
    alpha = args.alpha
    target_layers = args.octave_layers
    model_name = args.model

    MODELS = {
        'HED': lambda: HED(device),
        'OCTHED': lambda: OCTHED(device, alpha=float(alpha), octave_layers=target_layers),
        'EXITHED': lambda: EXITHED(device),
        'UNCERTHED': lambda: UncertHED(device),
        'FFCHED' : lambda: FFCHED(device, ratio = float(alpha), fourier_layer=target_layers),
        'DENSEHED' : lambda: DENSEHED(device),
        'UNCERTHEDOCT' : lambda: UNCERTHED_OCTAVE(device, float(alpha))
    }
    
    if model_name not in MODELS:
        raise ValueError(f"Can't recognize {model_name}.")
        
    return nn.DataParallel(MODELS[model_name]())