import torch as nn

class CurrentDiscriminatorModel():
    def __init__(self):
        super(CurrentDiscriminatorModel, self).__init__()
        pass

    def forward(self, x):
        # x - shape[B, C, H, W]
        return x