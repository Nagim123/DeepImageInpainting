import torch
import os

from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm

class Trainer:
    """
    Class for training image inpainting models.
    """

    def __init__(self, model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, loss_fn, optimizer: Optimizer, device: str, keep_original: bool = False) -> None:
        """
        Creates trainer.

        Parameters:
            model (Module): Torch model for training.
            train_loader (DataLoader): Training set dataloader.
            val_loader (DataLoader): Validation set dataloader.
            loss_fn (Function): Loss function.
            optimizer (Optimizer): Optimizer.
            device (str): Device to train on.
            keep_original (bool): Keep pixels between [0..255]
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.keep_original = keep_original

    def train_one_epoch(self) -> float:
        """
        One train cycle.

        Returns:
            float: Train loss.
        """

        self.model.train()
        running_loss = 0.0
        total = 0
        progress = tqdm(self.train_loader)
        for batch in progress:
            input, target = batch
            input, target = input.to(self.device), target.to(self.device)
            
            if self.keep_original:
                target = (target * 255).long()
            
            total += 1
            self.optimizer.zero_grad()
            reconstruction = self.model(input).to(self.device)
            loss = self.loss_fn(reconstruction, target)
            loss.backward()
            running_loss += loss.item()
            self.optimizer.step()
            progress.set_postfix({"loss": loss.item()})
        return running_loss / total

    def val_one_epoch(self) -> float:
        """
        One validation cycle.

        Returns:
            float: Validation loss.
        """
        
        with torch.no_grad():
            self.model.eval()
            running_loss = 0.0
            total = 0
            progress = tqdm(self.val_loader)
            for batch in progress:
                input, target = batch
                input, target = input.to(self.device), target.to(self.device)
                
                if self.keep_original:
                    target = (target * 255).long()

                total += 1
                reconstruction = self.model(input).to(self.device)
                loss = self.loss_fn(reconstruction, target)
                running_loss += loss.item()
                progress.set_postfix({"e_loss": loss.item()})
            return running_loss / total
    
    def train(self, epochs: int, save_path: str) -> None:
        """
        Do training of model for specified number of epochs.

        Parameters:
            epochs (int): How much epochs to train.
            save_path (str): Path where to save model.
        """
        best_loss = 1e9
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            train_loss = self.train_one_epoch()
            val_loss = self.val_one_epoch()
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(save_path, "best_weights.pt"))
            print(f"Average train loss:{train_loss} \n Average validation loss:{val_loss}")
            torch.save(self.model.state_dict(), os.path.join(save_path, "last_weights.pt"))