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

def train_one_epoch_with_discriminator(generator, discriminator, train_loader, loss_fn, opt_gen, opt_disc, device):
    generator.train()
    discriminator.train()
    running_loss = 0.0
    total = 0
    progress = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, batch in progress:
        input, target = batch
        input, target = input.to(device), target.to(device)
        # Generate fake image
        fake = generator(input).to(device)
        total += 1

        # Training Discriminator
        opt_disc.zero_grad()
        disc_real = discriminator(target).to(device).reshape(-1)
        loss_disc_real = loss_fn(disc_real, torch.ones_like(disc_real))
        disc_fake = discriminator(fake.detach()).to(device).reshape(-1)
        loss_disc_fake = loss_fn(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2

        
        loss_disc.backward()
        opt_disc.step()

        # Training Generator
        opt_gen.zero_grad()
        output = discriminator(fake).to(device).reshape(-1)
        loss_gen = (loss_fn(output, torch.ones_like(output)) + loss_fn(output, target))
        loss_gen.backward()
        opt_gen.step()
        running_loss += loss_gen.item()
        progress.set_postfix({"gen_loss": loss_gen.item(), "dicr_loss": loss_disc.item()})

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