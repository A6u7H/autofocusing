import torch.nn as nn

from torchvision.models import mobilenet_v3_large
from typing import Dict, Any


class MobileNetV3Large(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(MobileNetV3Large, self).__init__()

        self.config = config
        self.model = mobilenet_v3_large(
            pretrained=config.pretrained,
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
                16,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False
            )
        self.model.classifier[-1] = nn.Linear(config.input_dim, config.output_dim)

    def forward(self, x):
        return self.model(x)