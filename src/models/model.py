import torch.nn as nn

class CurrentModel(nn.Module):
    def __init__(self):
        super(CurrentModel, self).__init__()
        self.encode_layer = nn.Sequential(
            nn.Conv2d(3, 32, (3,3), (1,1), 1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, (3,3), (1,1), 1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(64, 128, (3,3), (1,1), 1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
        )
        self.hidden_layer = nn.Linear(2048, 1024)
        self.decode_layer = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (2,2), (2,2)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (2,2), (2,2)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, (2,2), (2,2)),
            nn.ReLU(),
        )

    def forward(self, x):
        # x - shape[B, C, H, W]
        x = self.encode_layer(x)
        x = x.view(-1, 2048)
        x = self.hidden_layer(x)
        x = x.view(-1, 64, 4, 4)
        x = self.decode_layer(x)
        return x
