import pytorch_lightning as pl
import os

from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader
from pytorch_lightning.trainer.supporters import CombinedLoader
from hydra.utils import instantiate

from dataset.utils import split_dataset


class DataModule(pl.LightningDataModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def prepare_data(self):
        search_pattern = "**/*.jpg"
        self.data_same_path = Path(self.config.dataset.same_data_path)
        self.data_diff_path = Path(self.config.dataset.diff_data_path)

        self.rgb_data = {}
        for segment_name in os.listdir(self.config.dataset.rgb_data_path):
            segment_path = os.path.join(
                self.config.dataset.rgb_data_path,
                segment_name
            )
            images = []
            for image_name in os.listdir(segment_path):
                image_path = os.path.join(segment_path, image_name)
                images.append(image_path)
            self.rgb_data[segment_name] = images

        self.same_data = [
            image_paph
            for image_paph in self.data_same_path.glob(search_pattern)
            if image_paph.is_file()
        ]
        self.diff_data = [
            image_paph
            for image_paph in self.data_diff_path.glob(search_pattern)
            if image_paph.is_file()
        ]

    def setup(self, stage: Optional[str] = None) -> None:
        train_transfrom = instantiate(self.config.dataset.train_transform)
        val_transfrom = instantiate(self.config.dataset.val_transform)
        test_transfrom = instantiate(self.config.dataset.test_transform)

        images_train_rgb, images_val_rgb = split_dataset(
            self.rgb_data,
            self.config.dataset.train_ratio
        )

        self.train_data_rgb = instantiate(
            self.config.dataset.train_dataset,
            images_data=images_train_rgb,
            transform=train_transfrom
        )

        self.val_data_rgb = instantiate(
            self.config.dataset.val_dataset,
            images_data=images_val_rgb,
            transform=val_transfrom
        )

        self.test_data_same = instantiate(
            self.config.dataset.test_dataset,
            images_data=self.same_data,
            transform=test_transfrom
        )

        self.test_data_diff = instantiate(
            self.config.dataset.test_dataset,
            images_data=self.diff_data,
            transform=test_transfrom
        )

    def train_dataloader(self) -> DataLoader:
        if self.config.dataset.train_dataloader._target_ is not None:
            return instantiate(
                self.config.dataset.train_dataloader,
                dataset=self.train_data_rgb
            )

    def val_dataloader(self) -> DataLoader:
        return instantiate(
            self.config.dataset.val_dataloader,
            dataset=self.val_data_rgb
        )

    def test_dataloader(self) -> DataLoader:
        val_loader_same = instantiate(
            self.config.dataset.test_dataloader,
            dataset=self.test_data_same
        )
        val_loader_diff = instantiate(
            self.config.dataset.test_dataloader,
            dataset=self.test_data_diff
        )
        loaders = {
            "same_protocol": val_loader_same,
            "diff_protocol": val_loader_diff
        }

        return CombinedLoader(loaders, mode="max_size_cycle")