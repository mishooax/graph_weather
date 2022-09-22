from typing import Tuple, List, Optional

import numpy as np
import torch
import pytorch_lightning as pl

from graph_weather.models.autoencoder import GraphWeatherAutoencoder
from graph_weather.models.losses import NormalizedMSELoss
from graph_weather.data.wb_datamodule import WeatherBenchDataBatch


class LitGraphAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        lat_lons: List,
        feature_dim: int,
        aux_dim: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        norm_type: Optional[str] = "LayerNorm",
    ) -> None:
        super().__init__()
        self.gnn = GraphWeatherAutoencoder(
            lat_lons,
            feature_dim=feature_dim,
            aux_dim=aux_dim,
            hidden_dim_decoder=hidden_dim,
            hidden_dim_processor_node=hidden_dim,
            hidden_layers_processor_edge=hidden_dim,
            hidden_dim_processor_edge=hidden_dim,
            norm_type=norm_type,
        )
        self.loss = NormalizedMSELoss(feature_variance=np.ones((feature_dim,)), lat_lons=lat_lons)
        self.feature_dim = feature_dim
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gnn(x)

    def training_step(self, batch: WeatherBenchDataBatch, batch_idx: int) -> torch.Tensor:
        del batch_idx  # not used
        x = batch.X[0]
        y_hat = self(x)  # prediction at rollout step rstep
        # we're reconstructing the input
        train_loss = self.loss(y_hat, x[..., : self.feature_dim])
        self.log("train_wmse", train_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True, batch_size=batch.X[0].shape[0])
        return train_loss

    def validation_step(self, batch: WeatherBenchDataBatch, batch_idx: int) -> torch.Tensor:
        val_loss = self._shared_eval_step(batch, batch_idx)
        self.log("val_wmse", val_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True, batch_size=batch.X[0].shape[0])
        return val_loss

    def _shared_eval_step(self, batch: WeatherBenchDataBatch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        with torch.no_grad():
            x = batch.X[0]
            y_hat = self(x)
            eval_loss = self.loss(y_hat, x[..., : self.feature_dim])
        return eval_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), betas=(0.0, 0.999), lr=self.lr)
