import albumentations as A
import torch
import cv2

from typing import Tuple
from albumentations.pytorch import ToTensorV2

from dataset.utils import get_fourier_channel


def img_to_patch(image, patch_size, flatten_channels=True):
    images = []
    for i in range(0, image.shape[0] - patch_size + 1, patch_size):
        for j in range(0, image.shape[1] - patch_size + 1, patch_size):
            images.append(image[i:i+patch_size, j:j+patch_size, :])
    return images


class TrainFocusingTransform:
    def __init__(
        self,
        mean: Tuple[float] = (0.485, 0.456, 0.406),
        std: Tuple[float] = (0.229, 0.224, 0.225),
        add_fourier: bool = False
    ) -> None:
        self.mean = mean
        self.std = std
        self.add_fourier = add_fourier

        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    def __call__(self, img):
        img = cv2.medianBlur(img, 3)
        img = self.transform(image=img)["image"]

        if self.add_fourier:
            magnitude_spectrum_tensor = get_fourier_channel(img)
            return torch.cat([
                img,
                magnitude_spectrum_tensor.unsqueeze(0)
            ], dim=0)
        else:
            return img


class ValFocusingTransform:
    def __init__(
        self,
        mean: Tuple[float] = (0.485, 0.456, 0.406),
        std: Tuple[float] = (0.229, 0.224, 0.225),
        add_fourier: bool = False
    ) -> None:
        self.mean = mean
        self.std = std
        self.add_fourier = add_fourier

        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    def __call__(self, img):
        img = cv2.medianBlur(img, 3)
        img = self.transform(image=img)["image"]

        if self.add_fourier:
            magnitude_spectrum_tensor = get_fourier_channel(img)
            return torch.cat([
                img,
                magnitude_spectrum_tensor.unsqueeze(0)
            ], dim=0)
        else:
            return img


class TestFocusingTransform:
    def __init__(
        self,
        mean: Tuple[float] = (0.485, 0.456, 0.406),
        std: Tuple[float] = (0.229, 0.224, 0.225),
        add_fourier: bool = False,
        crop_size: Tuple[int] = (2016, 2240)
    ) -> None:
        self.mean = mean
        self.std = std
        crop_height, crop_width = crop_size
        self.add_fourier = add_fourier

        self.post_transform = A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    def __call__(self, img):
        images = img_to_patch(img, 224)
        new_images = []
        for i in range(len(images)):
            img = cv2.medianBlur(images[i], 3)
            img = self.post_transform(image=img)["image"]
            new_images.append(img)
        if self.add_fourier:
            magnitude_spectrum_tensor = get_fourier_channel(img)
            return torch.cat([
                img,
                magnitude_spectrum_tensor.unsqueeze(0)
            ], dim=0)
        else:
            return torch.stack(new_images, dim=0)
