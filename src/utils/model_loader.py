from ..models import vanilla_ae, unet, pixel_cnn
import os
import torch

SUPPORTED_MODELS = {
    "vanilla_ae": vanilla_ae.VanillaAE(),
    "unet": unet.UNet(),
    "pixel_cnn": pixel_cnn.PixelCNN(),
}

def load_model(model_name: str, path_to_weights: str = None) -> torch.nn.Module:
    """
    Loads selected model.

    Parameters:
        path_to_model (str): Path to serialized model.
        path_to_weights (str): Path to weights of model (OPTIONAL).

    Returns:
        Module: Torch model.
    """

    if not model_name in get_available_models():
        raise Exception(f"Model is not found!")
    else:
        model = SUPPORTED_MODELS[model_name]
        if not path_to_weights is None:
            if not os.path.exists(path_to_weights):
                raise Exception("Model is loaded but weights not found!")
            model.load_state_dict(torch.load(path_to_weights))
    
    return model

def get_available_models() -> list[str]:
    """
    Return list of current available models.

    Returns:
        list[str]: List of names of current available models.
    """
    return list(SUPPORTED_MODELS.keys())