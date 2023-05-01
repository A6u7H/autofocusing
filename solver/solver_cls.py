import torch
import pytorch_lightning as pl

from omegaconf import DictConfig
from torch import Tensor
from hydra.utils import instantiate


class Solver(pl.LightningModule):
    def __init__(self, model, config: DictConfig) -> None:
        super(Solver, self).__init__()
        self.model = model
        self.config = config
        self.focus_loss = instantiate(self.config.optimizer.loss)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer = instantiate(
            self.config.optimizer.optimizer,
            params=params
        )

        scheduler = instantiate(
            self.config.optimizer.scheduler,
            optimizer=optimizer
        )

        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,
           'monitor': 'val/loss'
        }

    def loss_fn(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.focus_loss(pred, target)

    @torch.no_grad()
    def metric_fn(self, pred: Tensor, target: Tensor, mode: str = "train") -> Tensor:
        focus_metric = instantiate(self.config.optimizer.metrics)
        return focus_metric(pred, target, mode)

    def get_metrics(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = pred.view(-1)
        mae_value = torch.abs(pred.view(-1) - target).mean()
        mape_value = torch.abs(pred - target) / max(torch.abs(pred))
        return mae_value, mape_value

    def training_step(self, batch: Tensor, batch_idx: int):
        image, target_focus = batch
        pred_focus = self.model(image[0])
        loss = self.loss_fn(pred_focus, target_focus)
        metrics = self.metric_fn(pred_focus, target_focus)

        self.log("train/loss", loss)
        return {
            "loss": loss,
            "metrics": metrics
        }

    def training_epoch_end(self, outputs) -> None:
        accuracy = torch.tensor([
            (output['metrics']["correct_pred"], output['metrics']["total"])
            for output in outputs
        ])

        sum_stats = torch.sum(accuracy, dim=0)
        self.log("train/accuracy_mean", (sum_stats[0] / sum_stats[1]).item())

    def validation_step(self, batch: Tensor, batch_idx: int):
        image, target_focus = batch
        pred_focus = self.model(image[0])
        loss = self.loss_fn(pred_focus, target_focus)
        metrics = self.metric_fn(pred_focus, target_focus)
        self.log("val/loss", loss)

        return {
            "loss": loss,
            "metrics": metrics
        }

    def test_step(self, batch: Tensor, batch_idx: int):
        test_info = {}
        for key, batch_key in batch.items():
            images, target_focus = batch_key
            image_count = images[0].shape[1]
            predictions = torch.cat([
                self.model(images[0][:, i])
                for i in range(image_count)
            ], axis=0)

            loss = self.loss_fn(
                predictions,
                target_focus.repeat(predictions.shape[0])
            )
            metrics = self.metric_fn(predictions, target_focus, mode="test")

            test_info[key] = {"metrics": metrics, "loss": loss}
            self.log("test/loss", loss)

        return test_info

    def test_epoch_end(self, outputs) -> None:
        test_metrics = {
            "same_protocol": [],
            "diff_protocol": []
        }

        for output in outputs:
            test_metrics["same_protocol"].append((
                output["same_protocol"]["metrics"]["correct_pred"],
                output["same_protocol"]["metrics"]["total"]
            ))
            test_metrics["diff_protocol"].append((
                output["diff_protocol"]["metrics"]["correct_pred"],
                output["diff_protocol"]["metrics"]["total"]
            ))

        for key, metrics in test_metrics.items():
            sum_stats = torch.sum(torch.tensor(metrics), dim=0)
            self.log(
                f"test/{key}_correct_pred_mean",
                (sum_stats[0] / sum_stats[1]).item()
            )

    def validation_epoch_end(self, outputs) -> None:
        accuracy = torch.tensor([
            (output['metrics']["correct_pred"], output['metrics']["total"])
            for output in outputs
        ])
        sum_stats = torch.sum(accuracy, dim=0)
        self.log("val/accuracy_mean", (sum_stats[0] / sum_stats[1]).item())

    def fit(self):
        self.log_artifact(".hydra/config.yaml")
        self.log_artifact(".hydra/hydra.yaml")
        self.log_artifact(".hydra/overrides.yaml")
        self.trainer.fit(
            self,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader
        )

    def log_artifact(self, artifact_path: str):
        self.logger.experiment.log_artifact(self.logger.run_id, artifact_path)
