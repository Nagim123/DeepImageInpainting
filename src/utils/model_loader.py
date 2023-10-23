import os
import torch
from .constants import PATH_TO_MODELS

def load_model(path_to_model: str, path_to_weights: str = None) -> torch.Module:
    """
    Loads model from torch script file.

    Parameters:
        path_to_model (str): Path to serialized model.
        path_to_weights (str): Path to weights of model (OPTIONAL).

    Returns:
        Module: Torch model.
    """

    if not os.path.exists(path_to_model):
        raise Exception(f"Model is not found!")
    else:
        model = torch.jit.load(path_to_model)
        if not path_to_weights is None:
            if not os.path.exists(path_to_weights):
                raise Exception("Model is loaded but weights not found!")
            model.load_state_dict(torch.load(path_to_weights))
    
    return model
