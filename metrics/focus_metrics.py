import torch
import torch.nn as nn


class FocusMetrics(nn.Module):
    def __init__(self, config):
        super(FocusMetrics, self).__init__()
        self.config = config
        self.l1_loss = nn.L1Loss()

    def forward(self, prediction, target):
        l1_value = self.l1_loss(prediction.view(-1), target)
        return {
            "l1_loss": l1_value.item(),
        }


class FocusMetricsCls(nn.Module):
    def __init__(self, config):
        super(FocusMetricsCls, self).__init__()
        self.config = config

    def forward(self, prediction, target, mode="train"):
        class_pred = prediction.argmax(-1)
        if mode == "test":
            class_pred = torch.median(class_pred, dim=0, keepdim=True)[0]
        return {
            "correct_pred": (class_pred == target).sum().item(),
            "total": len(target)
        }


class FocusMetricsMulti(nn.Module):
    def __init__(self, config):
        super(FocusMetricsMulti, self).__init__()
        self.config = config
        self.l1_loss = nn.L1Loss()

    def forward(self, prediction_reg, prediction_cls, target_reg, target_cls, mode="train"):
        l1_value = self.l1_loss(prediction_reg.view(-1), target_reg)
        class_pred = prediction_cls.argmax(-1)
        if mode == "test":
            class_pred = torch.median(class_pred, dim=0, keepdim=True)[0]
        return {
            "correct_pred": (class_pred == target_cls).sum().item(),
            "l1_loss": l1_value.item(),
            "total": len(target_cls)
        }
