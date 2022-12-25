import albumentations as A
import numpy as np
import cv2

from typing import Tuple
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


def split_dataset(data, train_ratio: float=0.8):
    train_size = int(len(data) * train_ratio)
    data = np.random.permutation(data)
    return data[:train_size], data[train_size:]

class TrainFocusingTransform:
    def __init__(self, 
                mean: Tuple[float]=(0.485, 0.456, 0.406), 
                std: Tuple[float]=(0.229, 0.224, 0.225)
        ) -> None: 

        self.mean = mean
        self.std = std

        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Rotate(90, p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    def __call__(self, img):
        img = cv2.medianBlur(img, 3)
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
        img = cv2.medianBlur(img, 3)
        return self.transform(image=img)


class TestFocusingTransform:
    def __init__(self, 
                mean: Tuple[float]=(0.485, 0.456, 0.406), 
                std: Tuple[float]=(0.229, 0.224, 0.225),
                crop_size: Tuple[int]=(2016, 2016)
        ) -> None: 
        crop_height, crop_width = crop_size
        self.mean = mean
        self.std = std

        self.transform = A.Compose([
            A.CenterCrop(crop_height, crop_width),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    def __call__(self, img):
        img = cv2.medianBlur(img, 3)
        return self.transform(image=img)