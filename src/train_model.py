
import argparse
import torch
import logging
from data.create_dataset_base import MaskImageDataset
from models.loss_functions.bce_loss import bce_loss
from models.model import CurrentModel
from tqdm import tqdm
import pathlib
import os
script_path = pathlib.Path(__file__).parent.resolve()

def train_one_epoch(model, train_loader, loss_fn, optimizer):
    model.train()
    running_loss = 0.0
    total = 0
    progress = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, batch in progress:
        input, target = batch
        total += 1
        optimizer.zero_grad()
        reconstruction = model(input)
        loss = loss_fn(reconstruction, target)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        progress.set_postfix({"loss": loss.item()})
    return running_loss / total

def train_one_epoch_with_discriminator(generator, discriminator, train_loader, loss_fn, opt_gen, opt_disc):
    generator.train()
    discriminator.train()
    running_loss = 0.0
    total = 0
    progress = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, batch in progress:
        input, target = batch
        # Generate fake image
        fake = generator(input)
        total += 1

        # Training Discriminator
        disc_real = discriminator(target).reshape(-1)
        loss_disc_real = loss_fn(disc_real, torch.ones_like(disc_real))
        disc_fake = discriminator(fake).reshape(-1)
        loss_disc_fake = loss_fn(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2

        discriminator.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        # Training Generator
        ouput = disc_fake(fake).reshape(-1)
        loss_gen = loss_fn(ouput, torch.ones_like(ouput))
        generator.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        running_loss += loss_gen.item()
        progress.set_postfix({"loss": loss_gen.item()})

    return running_loss / total

def val_one_epoch(mode, val_loader, loss_fn):
    with torch.no_grad():
        mode.eval()
        running_loss = 0.0
        total = 0
        progress = tqdm(enumerate(val_loader), total=len(val_loader))
        for i, batch in progress:
            input, target = batch
            total += 1
            reconstruction = model(input)
            loss = loss_fn(reconstruction, target)
            running_loss += loss.item()
            progress.set_postfix({"e_loss": loss.item()})
        return running_loss / total

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
    path_to_model = os.path.join(script_path, f"models/checkpoints/{args.model_name}.pt")
    path_to_weights = os.path.join(script_path, f"models/checkpoints/{args.model_name}.pth")

    if not os.path.exists(path_to_model):
        logging.warn(f"Model {args.model_name} is not enough. Training current one from scratch!")
        model = CurrentModel()
        model_scripted = torch.jit.script(model)
        model_scripted.save(path_to_model)
    else:
        model = torch.jit.load(path_to_model)
        if args.weights:
            if not os.path.exists(path_to_weights):
                raise Exception("Model is loaded but weights not found!")
            model.load_state_dict(torch.load(path_to_weights))
    train_loader, val_loader = MaskImageDataset(from_file=args.dataset).pack_to_dataloaders(batch_size=32)
    loss_fn = loss_functions[args.loss]()
    optimizer = torch.optim.Adam(model.parameters())

    best_loss = 1e9
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = val_one_epoch(model, val_loader, loss_fn)
        if train_loss < best_loss:
            best_loss = train_loss
            logging.info("New best loss. Checkpoint is saved!")
            torch.save(model.state_dict(), path_to_weights)
        print(f"Epoch {epoch} train_loss:{train_loss}, val_loss:{val_loss}")