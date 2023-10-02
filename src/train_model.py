
import argparse
import torch
import logging
from data.create_dataset_base import MaskImageDataset
from models.loss_functions.bce_loss import bce_loss
from models.model import CurrentModel
from models.discriminator_model import CurrentDiscriminatorModel
from models.training_process import train_one_epoch, train_one_epoch_with_discriminator, val_one_epoch
import pathlib
import os
script_path = pathlib.Path(__file__).parent.resolve()

def load_model(model_name: str, require_weights: bool, default_variant):
    path_to_model = os.path.join(script_path, f"models/checkpoints/{model_name}.pt")
    path_to_weights = os.path.join(script_path, f"models/checkpoints/{model_name}.pth")

    if not os.path.exists(path_to_model):
        logging.warn(f"Model {model_name} is not enough. Training current one from scratch!")
        model = default_variant()
        model_scripted = torch.jit.script(model)
        model_scripted.save(path_to_model)
    else:
        model = torch.jit.load(path_to_model)
        if require_weights:
            if not os.path.exists(path_to_weights):
                raise Exception("Model is loaded but weights not found!")
            model.load_state_dict(torch.load(path_to_weights))
    
    return model, path_to_weights

loss_functions = {
    "BCELoss": bce_loss,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("model_name", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("epochs", type=int)
    parser.add_argument("loss", choices=list(loss_functions.keys()))
    parser.add_argument("--weights", action='store_true')
    parser.add_argument("--GAN_model", type=str)
    
    args = parser.parse_args()
    epochs = args.epochs
    

    # Loading Generator model
    model, model_weights_save_path = load_model(args.model_name, args.weights, CurrentModel)
    
    # Loading Discriminator model (if required)
    if args.GAN_model:
        discriminator, discriminator_save_path = load_model(args.model_name, args.weights, CurrentDiscriminatorModel)
        disc_optimizer = torch.optim.Adam(model.parameters())

    train_loader, val_loader = MaskImageDataset(from_file=args.dataset).pack_to_dataloaders(batch_size=32)
    loss_fn = loss_functions[args.loss]()
    optimizer = torch.optim.Adam(model.parameters())

    best_loss = 1e9
    for epoch in range(epochs):
        if args.GAN_model:
            train_loss = train_one_epoch_with_discriminator(model, discriminator, train_loader, loss_fn, optimizer, disc_optimizer)
        else:
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = val_one_epoch(model, val_loader, loss_fn)
        if train_loss < best_loss:
            best_loss = train_loss
            logging.info("New best loss. Checkpoint is saved!")
            torch.save(model.state_dict(), model_weights_save_path)
            if args.GAN_model:
                torch.save(discriminator.state_dict(), discriminator_save_path)
        print(f"Epoch {epoch} train_loss:{train_loss}, val_loss:{val_loss}")