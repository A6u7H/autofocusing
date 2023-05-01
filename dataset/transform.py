import numpy as np
import albumentations as A
import torch
import cv2

from typing import Tuple
from albumentations.pytorch import ToTensorV2

from dataset.utils import get_fourier_channel
from dataset.bisquet import get_besquet_features


def img_to_patch(image, patch_size):
    images = []
    for i in range(0, image.shape[0] - patch_size + 1, patch_size):
        row = []
        for j in range(0, image.shape[1] - patch_size + 1, patch_size):
            row.append(image[i:i+patch_size, j:j+patch_size, :])
        images.append(row)
    return np.array(images)


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
        self.besquet_transform = get_besquet_features

    def __call__(self, img):
        besquet_features = self.besquet_transform(img)

        img_blur = cv2.medianBlur(img, 3)
        img_transform = self.transform(image=img_blur)["image"]

        if self.add_fourier:
            magnitude_spectrum_tensor = get_fourier_channel(img)
            return torch.cat([
                img_transform,
                magnitude_spectrum_tensor.unsqueeze(0)
            ], dim=0), torch.tensor(besquet_features)
        return img_transform, torch.tensor(besquet_features)


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
        self.besquet_transform = get_besquet_features

    def __call__(self, img):
        besquet_features = self.besquet_transform(img)
        img_blur = cv2.medianBlur(img, 3)
        img_transform = self.transform(image=img_blur)["image"]

        if self.add_fourier:
            magnitude_spectrum_tensor = get_fourier_channel(img)
            return torch.cat([
                img_transform,
                magnitude_spectrum_tensor.unsqueeze(0)
            ], dim=0), torch.tensor(besquet_features)
        return img_transform, torch.tensor(besquet_features)


class TestFocusingTransform:
    def __init__(
        self,
        mean: Tuple[float] = (0.485, 0.456, 0.406),
        std: Tuple[float] = (0.229, 0.224, 0.225),
        add_fourier: bool = False,
        resize_size: Tuple[int] = (2016, 2240)
    ) -> None:
        self.mean = mean
        self.std = std
        crop_height, crop_width = resize_size
        self.add_fourier = add_fourier

        self.prep_transform = A.Compose([
            A.Resize(crop_height, crop_width),
        ])

        self.post_transform = A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
        self.besquet_transform = get_besquet_features

    def __call__(self, img):
        img_transform = self.prep_transform(image=img)["image"]
        images = img_to_patch(img_transform, 224)
        transformed_images = []
        besquet_features = []
        for i in range(images.shape[0]):
            row = []
            features_row = []
            for j in range(images.shape[1]):
                features = self.besquet_transform(img)
                features_row.append(torch.tensor(features))

                image = cv2.medianBlur(images[i][j], 3)
                image = self.post_transform(image=image)["image"]
                if self.add_fourier:
                    magnitude_spectrum_tensor = get_fourier_channel(images[i][j])
                    image = torch.cat([
                        image,
                        magnitude_spectrum_tensor.unsqueeze(0)
                    ], dim=0)
                row.append(image)
            besquet_features.append(torch.stack(features_row))
            transformed_images.append(torch.stack(row))
        besquet_features = torch.cat(besquet_features)
        transformed_images = torch.cat(transformed_images)
        return transformed_images, besquet_features


class TragetTransform:
    def __init__(
        self,
        task_type: str = "reg",
        num_segment: int = 10,
        label_range: Tuple[int] = (-10500, 10500)
    ) -> None:
        self.task_type = task_type
        self.num_segment = num_segment
        self.label_range = label_range

        len_range = label_range[1] - label_range[0]
        step = len_range // num_segment
        segments = [
            (i, i + step + 1)
            for i in range(label_range[0], label_range[1], step)
        ]
        self.segments2label = dict(zip(
            segments,
            [i for i in range(num_segment)]
        ))

    def __call__(self, defocus):
        if self.task_type == "reg":
            return torch.tensor(defocus/1000, dtype=torch.float32)
        elif self.task_type == "cls":
            if defocus < self.label_range[0]:
                return 0
            if defocus > self.label_range[1]:
                return self.num_segment - 1
            for segment, label in self.segments2label.items():
                if defocus >= segment[0] and defocus < segment[1]:
                    return label
