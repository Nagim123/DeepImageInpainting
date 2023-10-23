
import argparse
import torch
import logging
from data.create_dataset_base import MaskImageDataset
from models.loss_functions.bce_loss import bce_loss
from models.loss_functions.mean_square_error import mse_loss
from models.training_process import train_one_epoch, val_one_epoch

import pathlib
import os
script_path = pathlib.Path(__file__).parent.resolve()

def load_model(model_name: str, require_weights: bool):
    path_to_model = os.path.join(script_path, f"models/checkpoints/{model_name}.pt")
    path_to_weights = os.path.join(script_path, f"models/checkpoints/{model_name}.pth")

    if not os.path.exists(path_to_model):
        raise Exception(f"Model {model_name} is not found!")
    else:
        model = torch.jit.load(path_to_model)
        if require_weights:
            if not os.path.exists(path_to_weights):
                raise Exception("Model is loaded but weights not found!")
            model.load_state_dict(torch.load(path_to_weights))
    
    return model, path_to_weights

if __name__ == "__main__":

    loss_functions = {
        "BCELoss": bce_loss,
        "MSELoss": mse_loss,
    }

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("model_name", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("epochs", type=int)
    parser.add_argument("loss", choices=list(loss_functions.keys()))
    parser.add_argument("--weights", action='store_true')
    
    args = parser.parse_args()
    epochs = args.epochs
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Loading Generator model
    model, model_weights_save_path = load_model(args.model_name, args.weights)
    model = model.to(device)

    train_loader, val_loader = MaskImageDataset(from_file=args.dataset).pack_to_dataloaders(batch_size=32)
    loss_fn = loss_functions[args.loss]()
    optimizer = torch.optim.Adam(model.parameters())

    best_loss = 1e9
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = val_one_epoch(model, val_loader, loss_fn, device)
        if val_loss < best_loss:
            best_loss = val_loss
            logging.info("New best loss. Checkpoint is saved!")
            torch.save(model.state_dict(), model_weights_save_path)
        print(f"Epoch {epoch} train_loss:{train_loss}, val_loss:{val_loss}")