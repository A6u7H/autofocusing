import cv2
import numpy as np
import re

from torch.utils.data import Dataset
from typing import Optional, Any


class FocusingDataset(Dataset):
    def __init__(
        self,
        images_data: np.ndarray,
        pattern: str,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None
    ) -> None:
        self.pattern = pattern
        self.images_data = images_data

        self.transform = transform
        self.target_transform = target_transform

    def get_image_paths(self):
        search_pattern = "**/*.jpg"
        return [
            image_paph
            for image_paph in self.data_path.glob(search_pattern)
            if image_paph.is_file()
        ]

    def __getitem__(self, idx):
        image_path = str(self.images_data[idx])
        image_name = image_path.split("/")[-1]

        match = re.search(self.pattern, image_name)
        if match:
            if len(match.groups()) == 2:
                _, defocus = int(match.group(1)), int(match.group(2))
            else:
                defocus = int(match.group(1))
        image = cv2.imread(image_path)[..., ::-1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            defocus = self.target_transform(defocus)

        return image, defocus

    def __len__(self):
        return len(self.images_data)


class TwoImagesFocusingDataset(Dataset):
    def __init__(
        self,
        images_data: np.ndarray,
        pattern: str,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None
    ) -> None:
        self.pattern = pattern
        self.images_data = images_data

        self.transform = transform
        self.target_transform = target_transform

    def get_image_paths(self):
        search_pattern = "**/*.jpg"
        return [
            image_paph
            for image_paph in self.data_path.glob(search_pattern)
            if image_paph.is_file()
        ]

    def __getitem__(self, idx):
        image_path_one, image_path_two = self.images_data[idx]
        image_name_one = str(image_path_one).split("/")[-1]
        image_name_two = str(image_path_two).split("/")[-1]

        match_one = re.search(self.pattern, image_name_one)
        match_two = re.search(self.pattern, image_name_two)
        if match_one:
            if len(match_one.groups()) == 2:
                _, defocus_one = int(match_one.group(1)), int(match_one.group(2))
            else:
                defocus_one = int(match_one.group(1))
        if match_two:
            if len(match_two.groups()) == 2:
                _, defocus_two = int(match_two.group(1)), int(match_two.group(2))
            else:
                defocus_two = int(match_two.group(1))
        image_one = cv2.imread(image_path_one)[..., ::-1]
        image_two = cv2.imread(image_path_two)[..., ::-1]

        if self.transform:
            image_one, bisq_features_one = self.transform(image_one)
            image_two, bisq_features_two = self.transform(image_two)
            image = image_two - image_one
        if self.target_transform:
            defocus_one = self.target_transform(defocus_one)
            defocus_two = self.target_transform(defocus_two)
            defocus = defocus_two

        return (image, bisq_features_two), defocus

    def __len__(self):
        return len(self.images_data)
