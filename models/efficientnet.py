import torch.nn as nn

from torchvision.models import efficientnet_b7
from typing import Dict, Any


class EfficientNetLarge(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(EfficientNetLarge, self).__init__()

        self.config = config
        self.model = efficientnet_b7(
            progress=True,
            weights=config.weights
        )

        if self.config.freeze:
            for i, child in enumerate(self.model.children()):
                if i < 2:
                    for param in child.parameters():
                        param.requires_grad = False

        if config.change_first_layer:
            self.model.features[0][0] = nn.Conv2d(
                4,
                64,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False
            )
        self.model.classifier[-1] = nn.Linear(
            config.input_dim,
            config.output_dim
        )

    def forward(self, x, features=None):
        return self.model(x)


class EfficientNetLargeMulti(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(EfficientNetLargeMulti, self).__init__()

        self.config = config
        self.model = efficientnet_b7(
            progress=True,
            weights=config.weights
        )

        if self.config.freeze:
            for i, child in enumerate(self.model.children()):
                if i < 2:
                    for param in child.parameters():
                        param.requires_grad = False

        if config.change_first_layer:
            self.model.features[0][0] = nn.Conv2d(
                4,
                64,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False
            )
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(config.input_dim, config.output_dim_cls)
        )
        self.reg_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(config.input_dim, config.output_dim_reg)
        )

    def forward(self, x, features=None):
        features = self.model(x).view(-1, self.config.input_dim)
        return self.reg_head(features), self.cls_head(features)
