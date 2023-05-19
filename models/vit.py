import torch.nn as nn

from torchvision.models import vit_b_32
from typing import Dict, Any
from transformers import ViTModel


class VIT(nn.Module):

    def __init__(self, config: Dict[str, Any]):
        super(VIT, self).__init__()

        self.config = config
        # self.model = vit_b_32(weights=config.weights, progress=True, )
        self.model = ViTModel.from_pretrained(
            config.weights,
            add_pooling_layer=False
        )

        for param in self.model.encoder.parameters():
            param.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.model(x)['last_hidden_state']
        return self.head(features[:, 0, :])
