import os
import torch
from .constants import PATH_TO_MODELS

def load_model(model_name: str, require_weights: bool) -> torch.Module:
    """
    Loads model from torch script file.

    Parameters:
        model_name (str): Name of model to load.
        require_weights (bool): If true, then load weights to model.

    Returns:
        Module: Torch model.
    """
    path_to_model = os.path.join(PATH_TO_MODELS, model_name + ".pt")
    path_to_weights = os.path.join(PATH_TO_MODELS, model_name + ".pth")

    if not os.path.exists(path_to_model):
        raise Exception(f"Model {model_name} is not found!")
    else:
        model = torch.jit.load(path_to_model)
        if require_weights:
            if not os.path.exists(path_to_weights):
                raise Exception("Model is loaded but weights not found!")
            model.load_state_dict(torch.load(path_to_weights))
    
    return model
