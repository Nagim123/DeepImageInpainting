
import argparse
import torch
from data.create_dataset_base import MaskImageDataset
from torch.nn import BCELoss, MSELoss
from utils.trainer import Trainer
from utils.model_loader import load_model, get_available_models
from utils.constants import PATH_TO_MODELS

if __name__ == "__main__":
    
    # Define loss functions
    loss_functions = {
        "BCELoss": BCELoss(),
        "MSELoss": MSELoss(),
    }

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("model_name", choices=get_available_models())
    parser.add_argument("dataset", type=str)
    parser.add_argument("epochs", type=int)
    parser.add_argument("loss", choices=list(loss_functions.keys()))
    parser.add_argument("--weights", type=str)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model from torch script file
    model = load_model(args.model_name, args.weights)
    model = model.to(device)
    
    # Load dataset
    train_loader, val_loader = MaskImageDataset(from_file=args.dataset).pack_to_dataloaders(batch_size=32)
    
    # Set optimizer and chosen loss function
    loss_fn = loss_functions[args.loss]
    optimizer = torch.optim.Adam(model.parameters())

    # Initialize trainer and start training
    trainer = Trainer(model, train_loader, val_loader, loss_fn, optimizer, device)
    trainer.train(args.epochs, PATH_TO_MODELS)