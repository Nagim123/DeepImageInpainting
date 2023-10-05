import torch.nn as nn

class CurrentModel(nn.Module):
    def __init__(self):
        super(CurrentModel, self).__init__()
        self.encode_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3,3), (1,1), 1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(64, 256, (3,3), (1,1), 1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(256, 512, (3,3), (1,1), 1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(512, 1024, (3,3), (1,1), 1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
        )
        self.hidden_layer = nn.Linear(4096, 2048)
        self.decode_layer = nn.Sequential(
            nn.ConvTranspose2d(512, 256, (4,4), (2,2), (1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, (4,4), (2,2), (1,1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, (4,4), (2,2), (1,1)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x - shape[B, C, H, W]
        x = self.encode_layer(x)
        x = x.view(-1, 2048)
        x = self.hidden_layer(x)
        x = x.view(-1, 512, 2, 2)
        x = self.decode_layer(x)
        return x
