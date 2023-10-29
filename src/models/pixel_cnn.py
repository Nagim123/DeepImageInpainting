import torch
import torch.nn as nn

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class PixelCNN(nn.Module):
    def __init__(self):
        super(PixelCNN, self).__init__()
        self.net = nn.Sequential(
            MaskedConv2d('A', 3,  64, 7, 1, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 768, 1),
        )

    def forward(self, x, do_inference: bool = False):
        # x - shape[B, C, H, W]
        batch_size, channels, width, height = x.shape
        x = self.net(x)
        # net - shape [B, 768, H, W]
        x = x.reshape(batch_size, 256, channels, width, height)
        if do_inference:
            return torch.max(x, dim=1)[1].float()/255.0
        return x