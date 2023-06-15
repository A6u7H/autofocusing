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


class FocusLossMulti(nn.Module):
    def __init__(self, config):
        super(FocusLossMulti, self).__init__()
        self.config = config
        self.alpha = 0.8
        self.loss_reg_fn = nn.SmoothL1Loss(**self.config)
        self.loss_cls_fn = nn.CrossEntropyLoss()

    def forward(self, prediction_reg, prediction_cls, target_reg, target_cls):
        return self.alpha * self.loss_reg_fn(prediction_reg.view(-1), target_reg) + (1 - self.alpha) * self.loss_cls_fn(prediction_cls, target_cls)
