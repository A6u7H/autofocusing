import torch
import torch.nn as nn


class FocusMetrics(nn.Module):
    def __init__(self, config):
        super(FocusMetrics, self).__init__()
        self.config = config
        self.grid = torch.tensor(
            [i for i in range(config.min, config.max, config.step)]
        )
        self.l1_loss = nn.L1Loss()


    def forward(self, prediction, target):
        device = prediction.device

        l1_value = self.l1_loss(prediction.view(-1), target)
        repeated = prediction.repeat(1, len(self.grid))
        diff = abs(repeated - self.grid.to(device))
        class_pred = self.grid[diff.min(axis=1)[1]].to(device)
        return {
            "l1_loss": l1_value.item(),
            "correct_pred": (class_pred == target).sum().item(),
            "total": len(target)
        }

