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
        self.focus_metric = instantiate(self.config.optimizer.metrics)

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
    def metric_fn(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.focus_metric(pred, target)

    def training_step(self, batch: Tensor, batch_idx: int):
        image, target_focus_label, target_focus_continue = batch
        pred_focus_reg, pred_focus_cls = self.model(image[0])
        loss = self.focus_loss(pred_focus_reg, pred_focus_cls, target_focus_continue, target_focus_label)
        metrics = self.focus_metric(pred_focus_reg, pred_focus_cls, target_focus_continue, target_focus_label)

        self.log("train/loss", loss)
        return {
            "loss": loss,
            "metrics": metrics
        }

    def training_epoch_end(self, outputs) -> None:
        l1_losses = torch.tensor([
            output['metrics']["l1_loss"] for output in outputs
        ])
        accuracy = torch.tensor([
            (output['metrics']["correct_pred"], output['metrics']["total"])
            for output in outputs
        ])

        sum_stats = torch.sum(accuracy, dim=0)
        self.log("train/accuracy_mean", (sum_stats[0] / sum_stats[1]).item())
        self.log("train/l1_loss_mean", torch.mean(l1_losses))
        self.log("train/l1_loss_std", torch.std(l1_losses))

    def validation_step(self, batch: Tensor, batch_idx: int):
        image, target_focus_label, target_focus_continue = batch
        pred_focus_reg, pred_focus_cls = self.model(image[0])
        loss = self.focus_loss(pred_focus_reg, pred_focus_cls, target_focus_continue, target_focus_label)
        metrics = self.focus_metric(pred_focus_reg, pred_focus_cls, target_focus_continue, target_focus_label)
        self.log("val/loss", loss)

        return {
            "loss": loss,
            "metrics": metrics
        }

    def test_step(self, batch: Tensor, batch_idx: int):
        test_info = {}
        for key, batch_key in batch.items():
            images, target_focus_label, target_focus_continue = batch_key
            image_count = images[0].shape[1]
            prediction = [
                self.model(images[0][:, i]) #  images[1][:, i]
                for i in range(image_count)
            ]
            predictions_reg = torch.cat([pred[0] for pred in prediction], axis=1)
            pred_focus_cls = torch.cat([pred[1] for pred in prediction], axis=1)

            pred_focus_reg = torch.median(predictions_reg, dim=1, keepdim=True)[0]
            loss = self.focus_loss(pred_focus_reg, pred_focus_cls, target_focus_continue, target_focus_label)
            metrics = self.focus_metric(pred_focus_reg, pred_focus_cls, target_focus_continue, target_focus_label, mode="test")

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
                output["same_protocol"]["metrics"]["l1_loss"],
                output["same_protocol"]["metrics"]["correct_pred"],
                output["same_protocol"]["metrics"]["total"]
            ))
            test_metrics["diff_protocol"].append((
                output["diff_protocol"]["metrics"]["l1_loss"],
                output["same_protocol"]["metrics"]["correct_pred"],
                output["same_protocol"]["metrics"]["total"]
            ))

        for key, metrics in test_metrics.items():
            sum_stats = torch.sum(torch.tensor(metrics), dim=0)
            self.log(
                f"test/{key}_correct_pred_mean",
                (sum_stats[1] / sum_stats[2]).item()
            )
            self.log(
                f"test/{key}_l1_loss_mean",
                torch.mean(torch.tensor(metrics[0]))
            )
            self.log(
                f"test/{key}_l1_loss_std",
                torch.std(torch.tensor(metrics[0]))
            )

    def validation_epoch_end(self, outputs) -> None:
        l1_losses = torch.tensor([
            output['metrics']["l1_loss"] for output in outputs
        ])
        accuracy = torch.tensor([
            (output['metrics']["correct_pred"], output['metrics']["total"])
            for output in outputs
        ])

        sum_stats = torch.sum(accuracy, dim=0)
        self.log("val/accuracy_mean", (sum_stats[0] / sum_stats[1]).item())
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
