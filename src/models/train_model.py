
import argparse


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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("model")
    parser.add_argument("dataset")
    parser.add_argument("epochs")
    parser.add_argument("loss")
    
    args = parser.parse_args()
    epochs = args.epochs

    #train_loader = 

    for epoch in range(epochs):
        train_one_epoch()