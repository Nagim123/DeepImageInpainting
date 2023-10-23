import torch
from tqdm import tqdm

def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    total = 0
    progress = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, batch in progress:
        input, target = batch
        input, target = input.to(device), target.to(device)
        total += 1
        optimizer.zero_grad()
        reconstruction = model(input).to(device)
        loss = loss_fn(reconstruction, target)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        progress.set_postfix({"loss": loss.item()})
    return running_loss / total

def val_one_epoch(model, val_loader, loss_fn, device):
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        total = 0
        progress = tqdm(enumerate(val_loader), total=len(val_loader))
        for i, batch in progress:
            input, target = batch
            input, target = input.to(device), target.to(device)

            total += 1
            reconstruction = model(input).to(device)
            loss = loss_fn(reconstruction, target)
            running_loss += loss.item()
            progress.set_postfix({"e_loss": loss.item()})
        return running_loss / total