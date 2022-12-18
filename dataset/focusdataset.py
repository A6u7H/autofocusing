import torch
import cv2
import numpy as np
import re
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Any, Tuple


class FocusingDataset(Dataset):
    def __init__(self, 
                images_data: np.ndarray, 
                pattern: str,
                transform: Optional[Any]=None,
                target_transform: Optional[Any]=None) -> None:

        # self.data_path = Path(data_path)
        self.pattern = pattern
        # self.images_data = self.get_image_paths()
        self.images_data = images_data

        self.transform = transform
        self.target_transform = target_transform

    def get_image_paths(self):
        search_pattern = "**/*.jpg"
        return [image_paph for image_paph in self.data_path.glob(search_pattern) if image_paph.is_file()]

    def __getitem__(self, idx):
        image_path = str(self.images_data[idx])
        image_name = image_path.split("/")[-1]

        match = re.search(self.pattern, image_name)
        if match:
            if len(match.groups()) == 2:
                seg_num, defocus = int(match.group(1)), int(match.group(2))
            else:
                defocus = int(match.group(1))
        
        image = cv2.imread(image_path)[:,:,::-1]

        if self.transform:
            image = self.transform(image)["image"]
        if self.target_transform:
            defocus = self.target_transform(defocus)

        return image, torch.tensor(defocus/1000, dtype=torch.float32)

    def __len__(self):
        return len(self.images_data)