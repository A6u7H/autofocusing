import torch.nn as nn


class FocusLoss(nn.Module):
    def __init__(self, config):
        super(FocusLoss, self).__init__()
        self.config = config
        self.loss_fn = nn.SmoothL1Loss(**self.config)

    def forward(self, prediction, target):
        return self.loss_fn(prediction.view(-1), target)


class FocusLossCls(nn.Module):
    def __init__(self, config):
        super(FocusLossCls, self).__init__()
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, prediction, target):
        return self.loss_fn(prediction, target)
