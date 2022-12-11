import os
import hydra
import logging

from omegaconf import DictConfig
from hydra.utils import instantiate

from dataset.data_module import DataModule
from solver.solver import Solver


logger = logging.getLogger(__name__)

@hydra.main(config_path="config/", config_name="config.yaml", version_base="1.2.0")
def main(config: DictConfig) -> None:
    
    model = Solver(config)
    model_logger = instantiate(config.logger)
    datamodule = DataModule(config)
    trainer = instantiate(config.trainer, logger=model_logger)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()