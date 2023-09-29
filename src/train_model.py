
import argparse
import torch
import sys
import os
import pathlib
from data.create_dataset_base import MaskImageDataset
from models.loss_functions.bce_loss import bce_loss
from models.model import CurrentModel

def train_one_epoch(model, train_loader, loss_fn, optimizer):
    model.train()
    running_loss = 0.0
    total = 0
    for i, batch in enumerate(train_loader):
        input, target = batch
        total += input.shape[0]
        optimizer.zero_grad()
        reconstruction = model(input)
        loss = loss_fn(reconstruction, target)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    return running_loss / total

loss_functions = {
    "BCELoss": bce_loss,
}

if __name__ == "__main__":
    print("meow!")
    # parser = argparse.ArgumentParser(description="")
    # parser.add_argument("model")
    # parser.add_argument("dataset")
    # parser.add_argument("epochs")
    # parser.add_argument("loss", choices=list(loss_functions.keys()))
    
    # args = parser.parse_args()
    # epochs = args.epochs

    # if args.model is None:
    #     model = CurrentModel()
    # else:
    #     model = torch.load()
    # train_loader, _ = MaskImageDataset(from_file="cifar10.pt").pack_to_dataloaders(batch_size=32)
    # loss_fn = loss_functions[args.loss]
    # optimizer = torch.optim.Adam(model.parameters())

    # for epoch in range(epochs):
    #     train_one_epoch(model, train_loader, loss_fn, )