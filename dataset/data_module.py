from typing import Optional
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from pytorch_lightning.trainer.supporters import CombinedLoader
from hydra.utils import instantiate

from .utils import split_dataset


class DataModule(pl.LightningDataModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        train_transfrom=instantiate(self.config.dataset.train_transform)
        val_transfrom=instantiate(self.config.dataset.val_transform)

        train_data = instantiate(
            self.config.dataset.train_dataset, 
            transform=train_transfrom
        )

        self.train_data_rgb, self.val_data_rgb = split_dataset(train_data)

        self.val_data_same = instantiate(
            self.config.dataset.val_dataset_same, 
            transform=val_transfrom
        )

        self.val_data_diff = instantiate(
            self.config.dataset.val_dataset_diff, 
            transform=val_transfrom
        )

    def train_dataloader(self) -> DataLoader:
        if self.config.dataset.train_dataloader._target_ is not None:
            return instantiate(
                self.config.dataset.train_dataloader, dataset=self.train_data_rgb)

    def val_dataloader(self) -> DataLoader:
        val_loader_same = instantiate(self.config.dataset.val_dataloader, dataset=self.val_data_same)
        val_loader_diff = instantiate(self.config.dataset.val_dataloader, dataset=self.val_data_diff)
        val_loader_rgb = instantiate(self.config.dataset.val_dataloader, dataset=self.val_data_rgb)
        loaders = {
            "rgb": val_loader_rgb,
            "same_protocol": val_loader_same,
            "diff_protocol": val_loader_diff 
        }

        return CombinedLoader(loaders, mode="max_size_cycle")