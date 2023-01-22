import torch.nn as nn

from torchvision.models import vit_b_32
from typing import Dict, Any


class VIT(nn.Module):

    def __init__(self, config: Dict[str, Any]):
        super(VIT, self).__init__()

        self.config = config
        self.model = vit_b_32(weights=config.weights, progress=True)
        self.model.heads[-1] = nn.Linear(config.input_dim, config.output_dim)

    def forward(self, x):
        return self.model(x)
