import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from omegaconf import DictConfig
from torch import Tensor
from hydra.utils import instantiate


class Solver(pl.LightningModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        self.model = self.create_model()

    def forward(self, x):
        return self.model(x)

    def create_model(self):
        model = instantiate(self.config.model)
        return model

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer = instantiate(self.config.optimizer.optimizer, params=params)
        scheduler = instantiate(self.config.optimizer.scheduler, optimizer=optimizer)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,
           'monitor': 'same_protocol_val_loss'
        }
    
    def loss_fn(self, pred: Tensor, target: Tensor) -> Tensor:
        focus_loss = instantiate(self.config.optimizer.loss)
        return focus_loss(pred, target)

    @torch.no_grad()
    def metric_fn(self, pred: Tensor, target: Tensor) -> Tensor:
        focus_metric = instantiate(self.config.optimizer.metrics)
        return focus_metric(pred, target)

    def get_metrics(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = pred.view(-1)
        mae_value = torch.abs(pred.view(-1) - target).mean()
        mape_value = torch.abs(pred - target) / max(torch.abs(pred))
        return mae_value,mape_value

    def training_step(self, batch: Tensor, batch_idx: int):
        image, target_focus = batch
        pred_focus = self.model(image)
        loss = self.loss_fn(pred_focus, target_focus)
        metric = self.metric_fn(pred_focus, target_focus)

        self.log("train/loss", loss)
        return {
            "loss": loss,
            "metrics": metric
        }

    def training_epoch_end(self, outputs) -> None:
        l1_loss_total = 0
        total_correct = 0
        total_number = 0
        for output in outputs:
            l1_loss_total += output['metrics']["l1_loss"]
            total_correct += output['metrics']["correct_pred"]
            total_number += output['metrics']["total"]
        train_accuracy = total_correct / total_number
        l1_loss_total = l1_loss_total / len(outputs)
        self.log("train/accuracy", train_accuracy)
        self.log("train/l1_loss", l1_loss_total)

    def validation_step(self, batch: Tensor, batch_idx: int):
        dataset_dict = {}
        for key, value in batch.items():
            image, target_focus = value
            pred_focus = self.model(image)
            loss = self.loss_fn(pred_focus, target_focus)
            metrics = self.metric_fn(pred_focus, target_focus)
            self.log(f"{key}_val/loss", loss)
            dataset_dict[key] = metrics

        return {
            "loss": loss,
            "metrics": dataset_dict
        }

    def validation_epoch_end(self, outputs) -> None:
        for output in outputs:
            accuracy_dict = {}
            for k, v in output['metrics'].items():
                if k not in accuracy_dict:
                    accuracy_dict[k] = [0, 0, 0]
                accuracy_dict[k][0] += v["l1_loss"]
                accuracy_dict[k][1] += v["correct_pred"]
                accuracy_dict[k][2] += v["total"]
        for k, v in accuracy_dict.items():
            self.log(f"{k}_val/accuracy", v[1] / v[2])
            self.log(f"{k}_val/l1_loss", v[0] / len(outputs))


    def fit(self):
        self.log_artifact(".hydra/config.yaml")
        self.log_artifact(".hydra/hydra.yaml")
        self.log_artifact(".hydra/overrides.yaml")
        self.trainer.fit(self, train_dataloaders=self.train_dataloader, val_dataloaders=self.val_dataloader)

    def log_artifact(self, artifact_path: str):
        self.logger.experiment.log_artifact(self.logger.run_id, artifact_path)