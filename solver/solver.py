import torch
import pytorch_lightning as pl

from omegaconf import DictConfig
from torch import Tensor
from hydra.utils import instantiate


class Solver(pl.LightningModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        self.focus_loss = instantiate(self.config.optimizer.loss)
        self.model = self.create_model()

    def forward(self, x):
        return self.model(x)

    def create_model(self):
        model = instantiate(self.config.model)
        return model

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
    def metric_fn(self, pred: Tensor, target: Tensor) -> Tensor:
        focus_metric = instantiate(self.config.optimizer.metrics)
        return focus_metric(pred, target)

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
        l1_losses = torch.tensor([
            output['metrics']["l1_loss"] for output in outputs
        ])
        self.log("train/l1_loss_mean", torch.mean(l1_losses))
        self.log("train/l1_loss_std", torch.std(l1_losses))

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
                self.model(images[0][:, i]) #  images[1][:, i]
                for i in range(image_count)
            ], axis=1)

            pred_focus = torch.median(predictions, dim=1, keepdim=True)[0]
            loss = self.loss_fn(pred_focus, target_focus)
            metrics = self.metric_fn(pred_focus, target_focus)

            test_info[key] = {"metrics": metrics, "loss": loss}
            self.log("test/loss", loss)

        return test_info

    def test_epoch_end(self, outputs) -> None:
        test_metrics = {
            "same_protocol": [],
            "diff_protocol": []
        }

        for output in outputs:
            test_metrics["same_protocol"].append(
                output["same_protocol"]["metrics"]["l1_loss"]
            )
            test_metrics["diff_protocol"].append(
                output["diff_protocol"]["metrics"]["l1_loss"]
            )

        for key, metrics in test_metrics.items():
            self.log(
                f"test/{key}_l1_loss_mean",
                torch.mean(torch.tensor(metrics))
            )
            self.log(
                f"test/{key}_l1_loss_std",
                torch.std(torch.tensor(metrics))
            )

    def validation_epoch_end(self, outputs) -> None:
        l1_losses = torch.tensor([output['metrics']["l1_loss"] for output in outputs])
        self.log("val/l1_loss_mean", torch.mean(l1_losses))
        self.log("val/l1_loss_std", torch.std(l1_losses))

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
