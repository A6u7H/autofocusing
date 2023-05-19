import pytorch_lightning as pl
import numpy as np
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

    @staticmethod
    def pair_dataset(data):
        seg2path = {}
        for path in data:
            # name = path.name
            # defocus = int(name[7:-4])
            seg_block = os.path.split(path)[0].split("/")[-1]
            if seg_block not in seg2path:
                seg2path[seg_block] = []
            seg2path[seg_block].append(path)
        test_data = []
        for _, images in seg2path.items():
            defocus = list(map(
                lambda x: int(x.name[7:-4]),
                images
            ))
            defocus2id = dict(zip(defocus, np.arange(len(defocus))))
            for cur_defocus, idx in defocus2id.items():
                delta_defocus = cur_defocus + 2000
                if delta_defocus in defocus2id:
                    new_idx = defocus2id[delta_defocus]
                    test_data.append((images[idx], images[new_idx]))
        return test_data


    def prepare_data(self):
        search_pattern = "**/*.jpg"
        self.data_same_path = Path(self.config.dataset.same_data_path)
        self.data_diff_path = Path(self.config.dataset.diff_data_path)

        self.rgb_data = {}
        for segment_name in os.listdir(self.config.dataset.data_path):
            segment_path = os.path.join(
                self.config.dataset.data_path,
                segment_name
            )
            images = []
            for image_name in os.listdir(segment_path):
                image_path = os.path.join(segment_path, image_name)
                images.append(image_path)
            self.rgb_data[segment_name] = images

        self.same_data = [
            image_path
            for image_path in self.data_same_path.glob(search_pattern)
            if image_path.is_file()
        ]
        if self.config.dataset.two_image_pipeline:
            self.same_data = DataModule.pair_dataset(self.same_data)

        self.diff_data = [
            image_path
            for image_path in self.data_diff_path.glob(search_pattern)
            if image_path.is_file()
        ]
        if self.config.dataset.two_image_pipeline:
            self.diff_data = DataModule.pair_dataset(self.diff_data)

    def setup(self, stage: Optional[str] = None) -> None:
        train_transfrom = instantiate(self.config.dataset.train_transform)
        val_transfrom = instantiate(self.config.dataset.val_transform)
        test_transfrom = instantiate(self.config.dataset.test_transform)
        target_transsform = instantiate(self.config.dataset.target_transform)

        images_train_rgb, images_val_rgb = split_dataset(
            self.rgb_data,
            self.config.dataset.train_ratio,
            self.config.dataset.smart_split,
            self.config.dataset.two_image_pipeline
        )

        self.train_data_rgb = instantiate(
            self.config.dataset.train_dataset,
            images_data=images_train_rgb,
            transform=train_transfrom,
            target_transform=target_transsform
        )

        self.val_data_rgb = instantiate(
            self.config.dataset.val_dataset,
            images_data=images_val_rgb,
            transform=val_transfrom,
            target_transform=target_transsform
        )

        self.test_data_same = instantiate(
            self.config.dataset.test_dataset,
            images_data=self.same_data,
            transform=test_transfrom,
            target_transform=target_transsform
        )

        self.test_data_diff = instantiate(
            self.config.dataset.test_dataset,
            images_data=self.diff_data,
            transform=test_transfrom,
            target_transform=target_transsform
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
