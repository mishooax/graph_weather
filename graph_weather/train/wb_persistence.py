from typing import Tuple, List

import numpy as np
import torch
import pytorch_lightning as pl

from graph_weather.models.losses import NormalizedMSELoss
from graph_weather.data.wb_datamodule import WeatherBenchDataBatch
from graph_weather.train.utils import build_pl_logger, get_args
from graph_weather.utils.config import YAMLConfig
from graph_weather.utils.logger import get_logger
from graph_weather.data.wb_datamodule import WeatherBenchTrainingDataModule

LOGGER = get_logger(__name__)


class LitPersistenceForecaster(pl.LightningModule):
    """
    Dummy Trainer that implements atmospheric persistence (forecasts that target == input).
    Use this to produce a simple baseline value for the "real" training / validation loss.
    """

    def __init__(
        self,
        lat_lons: List,
        feature_dim: int,
        rollout: int = 1,
    ) -> None:
        super().__init__()
        self.loss = NormalizedMSELoss(feature_variance=np.ones((feature_dim,)), lat_lons=lat_lons)
        self.feature_dim = feature_dim
        self.rollout = rollout
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Persistence: returns (a subset of) x"""
        return x[..., : self.feature_dim]

    def training_step(self, batch: WeatherBenchDataBatch, batch_idx: int) -> None:
        del batch, batch_idx  # not used
        raise NotImplementedError

    def validation_step(self, batch: WeatherBenchDataBatch, batch_idx: int) -> torch.Tensor:
        val_loss = self._shared_eval_step(batch, batch_idx)
        self.log("val_wmse", val_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True, batch_size=batch.X[0].shape[0])
        return val_loss

    def test_step(self, batch: WeatherBenchDataBatch, batch_idx: int) -> torch.Tensor:
        test_loss = self._shared_eval_step(batch, batch_idx)
        self.log("test_wmse", test_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True, batch_size=batch.X[0].shape[0])
        return test_loss

    def _shared_eval_step(self, batch: WeatherBenchDataBatch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        loss = torch.zeros(1, dtype=batch.X[0].dtype, device=self.device, requires_grad=False)
        with torch.no_grad():
            # start rollout
            x = batch.X[0]
            for rstep in range(self.rollout):
                y_hat = self(x)
                y = batch.X[rstep + 1]
                loss += self.loss(y_hat, y[..., : self.feature_dim])
        return loss

    def configure_optimizers(self):
        return None


def persistence(config: YAMLConfig) -> None:
    """
    Train entry point.
    Args:
        config: job configuration
    """
    # create data module (data loaders and data sets)
    dmod = WeatherBenchTrainingDataModule(config)

    # number of variables (features)
    num_features = dmod.ds_train.nlev * dmod.ds_train.nvar

    LOGGER.debug("Number of variables: %d", num_features)
    LOGGER.debug("Number of auxiliary (time-independent) variables: %d", dmod.const_data.nconst)

    model = LitPersistenceForecaster(
        lat_lons=dmod.const_data.latlons,
        feature_dim=num_features,
        rollout=config["model:rollout"],
    )

    trainer = pl.Trainer(
        accelerator="gpu",  # can run on the CPU, too - but it'll be slower (the GPU will speed up the loss computations)
        detect_anomaly=False,
        strategy=None,
        devices=1,
        num_nodes=1,
        precision=config["model:precision"],
        max_epochs=config["model:max-epochs"],
        logger=build_pl_logger(config),
        log_every_n_steps=config["output:logging:log-interval"],
        limit_train_batches=config["model:limit-batches:training"],
        limit_val_batches=config["model:limit-batches:validation"],
    )

    trainer.validate(model, datamodule=dmod)

    LOGGER.debug("---- DONE. ----")


def main() -> None:
    """Entry point for training."""
    args = get_args()
    config = YAMLConfig(args.config)
    persistence(config)
