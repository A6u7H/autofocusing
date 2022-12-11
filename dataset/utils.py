import albumentations as A
import torch

from typing import Tuple
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


def split_dataset(dataset: Dataset, ):
    train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
    return train_set, val_set

class TrainFocusingTransform:
    def __init__(self, 
                mean: Tuple[float]=(0.485, 0.456, 0.406), 
                std: Tuple[float]=(0.229, 0.224, 0.225)
        ) -> None: 

        self.mean = mean
        self.std = std

        self.transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5), # A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.Rotate(90, p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    def __call__(self, img):
        return self.transform(image=img)


class ValFocusingTransform:
    def __init__(self, 
                mean: Tuple[float]=(0.485, 0.456, 0.406), 
                std: Tuple[float]=(0.229, 0.224, 0.225)
        ) -> None: 

        self.mean = mean
        self.std = std

        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    def __call__(self, img):
        return self.transform(image=img)