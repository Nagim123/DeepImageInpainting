import torch.nn as nn

class SimpleAE(nn.Module):
    def __init__(self):
        super(SimpleAE, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, (3,3), (1,1), 1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(3, 32, (3,3), (1,1), 1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(64, 128, (3,3), (1,1), 1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(128, 256, (3,3), (1,1), 1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
        )
        self.deconv_layer == nn.Sequential(
            nn.ConvTranspose2d(256, 128, (2,2), (2,2), "same"),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, (2,2), (2,2), "same"),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (2,2), (2,2), "same"),
            nn.ReLU(),
        )

    def forward(self, x):
        pass