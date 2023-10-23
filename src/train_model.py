
import argparse
import torch
import os
from data.create_dataset_base import MaskImageDataset
from torch.nn import BCELoss, MSELoss
from utils.trainer import Trainer
from utils.model_loader import load_model

if __name__ == "__main__":

    loss_functions = {
        "BCELoss": BCELoss(),
        "MSELoss": MSELoss(),
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
    model = load_model(args.model_name, args.weights)
    model = model.to(device)

    train_loader, val_loader = MaskImageDataset(from_file=args.dataset).pack_to_dataloaders(batch_size=32)
    loss_fn = loss_functions[args.loss]
    optimizer = torch.optim.Adam(model.parameters())

    trainer = Trainer(model, train_loader, val_loader, loss_fn, optimizer, device)

    trainer.train(args.epochs, os.path.join())