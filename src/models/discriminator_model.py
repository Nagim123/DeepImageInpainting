import torch.nn as nn

class CurrentDiscriminatorModel(nn.Module):
    def __init__(self):
        super(CurrentDiscriminatorModel, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, (4,4), (2,2), 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, (4,4), (2,2), 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, (4,4), (2,2), 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, (4,4), (2,2), 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, (4,4), (2,2), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x - shape[B, C, H, W]
        x = self.conv_layer(x)
        return x