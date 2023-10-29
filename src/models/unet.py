import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # todo
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.downStep1 = downStep(3, 64, firstLayer=True)
        self.downStep2 = downStep(64, 128)
        self.downStep3 = downStep(128, 256)
        self.downStep4 = downStep(256, 512)
        self.downStep5 = downStep(512, 1024)

        self.upStep1 = upStep(1024, 512)
        self.upStep2 = upStep(512, 256)
        self.upStep3 = upStep(256, 128)
        self.upStep4 = upStep(128, 64)

        self.conv = nn.Conv2d(64, 3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # todo
        x1 = self.downStep1(x)
        # print('x1', x1.shape)
        x2 = self.downStep2(x1)
        # print('x2', x2.shape)
        x3 = self.downStep3(x2)
        # print('x3', x3.shape)
        x4 = self.downStep4(x3)
        # print('x4', x4.shape)
        x5 = self.downStep5(x4)
        # print('x5', x5.shape)

        x = self.upStep1(x5, x4)
        # print(x.shape)
        x = self.upStep2(x, x3)
        # print(x.shape)
        x = self.upStep3(x, x2)
        # print(x.shape)
        x = self.upStep4(x, x1)
        # print(x.shape)

        x = self.conv(x)
        # print(x.shape)
        x = self.sigmoid(x)

        return x

class downStep(nn.Module):
    def __init__(self, inC, outC, firstLayer=False):
        super(downStep, self).__init__()
        # todo
        self.firstLayer = firstLayer

        self.conv = nn.Sequential(
            nn.Conv2d(inC, outC, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(outC, outC, 3, padding=1),
            nn.ReLU())

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        # todo
        if not self.firstLayer:
            x = self.maxpool(x)

        x = self.conv(x)

        return x

class upStep(nn.Module):
    def __init__(self, inC, outC):
        super(upStep, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(inC, outC, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(outC, outC, 3, padding=1),
            nn.ReLU())

        self.upsampling = nn.ConvTranspose2d(inC, outC, 2, 2)

    def forward(self, x, x_down):
        # todo
        x = self.upsampling(x)

        x = torch.cat([x_down, x], dim=1)

        x = self.conv(x)

        return x