from typing import Tuple, List

from einops import rearrange
import numpy as np
import pytorch_lightning as pl
import torch

from graph_weather.models.losses import NormalizedMSELoss
from graph_weather.data.wb_datamodule import WeatherBenchDataBatch
from graph_weather.models.layers.unet import UNet
from graph_weather.utils.constants import _WB_LAT, _WB_LON


class LitUnetForecaster(pl.LightningModule):
    def __init__(
        self,
        lat_lons: List,
        feature_dim: int,
        aux_dim: int,
        lr: float = 1e-3,
        rollout: int = 1,
    ) -> None:
        super().__init__()
        self.unet = UNet(num_inputs=feature_dim + aux_dim, num_outputs=feature_dim, bilinear=True)
        self.loss = NormalizedMSELoss(feature_variance=np.ones((feature_dim,)), lat_lons=lat_lons)
        self.feature_dim = feature_dim
        self.lr = lr
        self.rollout = rollout
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)

    def training_step(self, batch: WeatherBenchDataBatch, batch_idx: int) -> torch.Tensor:
        del batch_idx  # not used
        assert len(batch.X) == (self.rollout + 1), "Rollout window doesn't match len(batch)!"
        train_loss = torch.zeros(1, dtype=batch.X[0].dtype, device=self.device, requires_grad=False)
        # start rollout
        x = batch.X[0]
        for rstep in range(self.rollout):
            # rearrange to the shape that the UNet expects
            x_ = rearrange(x, "b (h w) c -> b c h w", h=_WB_LAT, w=_WB_LON)
            # model prediction at rollout step rstep, then reshape back
            y_hat = rearrange(self(x_), "b c h w -> b (h w) c")
            # target
            y = batch.X[rstep + 1]
            # y includes the auxiliary variables, so we must leave those out when computing the loss
            train_loss += self.loss(y_hat, y[..., : self.feature_dim])
            # autoregressive predictions - we re-init the "variable" part of x
            x[..., : self.feature_dim] = y_hat
        self.log("train_wmse", train_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True, batch_size=batch.X[0].shape[0])
        return train_loss

    def validation_step(self, batch: WeatherBenchDataBatch, batch_idx: int) -> torch.Tensor:
        val_loss = self._shared_eval_step(batch, batch_idx)
        self.log("val_wmse", val_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True, batch_size=batch.X[0].shape[0])
        return val_loss

    def test_step(self, batch: WeatherBenchDataBatch, batch_idx: int) -> torch.Tensor:
        test_loss = self._shared_eval_step(batch, batch_idx)
        self.log("test_wmse", test_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True, batch_size=batch.X[0].shape[0])
        return test_loss

    def predict_step(self, batch: WeatherBenchDataBatch, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        del batch_idx  # not used
        preds: List[torch.Tensor] = []
        with torch.no_grad():
            # start rollout
            x = batch.X[0]
            for _ in range(self.rollout):
                x_ = rearrange(x, "b (h w) c -> b c h w", h=_WB_LAT, w=_WB_LON)
                y_hat = rearrange(self(x_), "b c h w -> b (h w) c")
                x[..., : self.feature_dim] = y_hat
                preds.append(y_hat)
        return torch.stack(preds, dim=-1), batch.idx  # stack along new last dimension, return sample indices too

    def _shared_eval_step(self, batch: WeatherBenchDataBatch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        loss = torch.zeros(1, dtype=batch.X[0].dtype, device=self.device, requires_grad=False)
        with torch.no_grad():
            # start rollout
            x = batch.X[0]
            for rstep in range(self.rollout):
                x_ = rearrange(x, "b (h w) c -> b c h w", h=_WB_LAT, w=_WB_LON)
                y_hat = rearrange(self(x_), "b c h w -> b (h w) c")
                y = batch.X[rstep + 1]
                loss += self.loss(y_hat, y[..., : self.feature_dim])
                x[..., : self.feature_dim] = y_hat
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), betas=(0.0, 0.999), lr=self.lr)
