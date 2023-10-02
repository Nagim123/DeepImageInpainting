import torch.nn as nn

class CurrentDiscriminatorModel():
    def __init__(self):
        super(CurrentDiscriminatorModel, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, (3,3), (2,2), 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, (3,3), (2,2), 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, (3,3), (2,2), 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, (3,3), (2,2), 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, (3,3), (2,2), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x - shape[B, C, H, W]
        x = self.conv_layer(x)
        return x