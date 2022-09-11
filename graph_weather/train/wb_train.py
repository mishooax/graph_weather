# Train a GNN model on the WeatherBench dataset
import datetime as dt
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from graph_weather.utils.config import YAMLConfig
from graph_weather.data.wb_datamodule import WeatherBenchTrainingDataModule
from graph_weather.utils.logger import get_logger
from graph_weather.train.wb_trainer import LitGraphForecaster
from graph_weather.train.wb_persistence import LitPersistenceForecaster
from graph_weather.train.utils import build_pl_logger, get_args

LOGGER = get_logger(__name__)


def train(config: YAMLConfig, persistence: bool = False) -> None:
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

    if not persistence:
        model = LitGraphForecaster(
            lat_lons=dmod.const_data.latlons,
            feature_dim=num_features,
            aux_dim=dmod.const_data.nconst,
            hidden_dim=config["model:hidden-dim"],
            num_blocks=config["model:num-blocks"],
            lr=config["model:learn-rate"],
            rollout=config["model:rollout"],
        )
    else:
        model = LitPersistenceForecaster(
            lat_lons=dmod.const_data.latlons,
            feature_dim=num_features,
            rollout=config["model:rollout"],
        )

    trainer = pl.Trainer(
        accelerator="gpu",
        callbacks=[
            EarlyStopping(monitor="val_wmse", min_delta=0.0, patience=3, verbose=False, mode="min"),
            ModelCheckpoint(
                dirpath=os.path.join(
                    config["output:basedir"],
                    config["output:checkpoints:ckpt-dir"],
                    dt.datetime.now().strftime("%Y%m%d_%H%M"),
                ),
                filename=config[f"output:model:checkpoint-filename"],
                monitor="val_wmse",
                verbose=False,
                save_top_k=config["output:model:save-top-k"],
                save_weights_only=True,
                mode="min",
                auto_insert_metric_name=True,
                save_on_train_epoch_end=True,
                every_n_epochs=1,
            ),
        ],
        detect_anomaly=config["model:debug:anomaly-detection"],
        strategy=config["model:strategy"],
        devices=config["model:num-gpus"],
        num_nodes=config["model:num-nodes"],
        precision=config["model:precision"],
        max_epochs=config["model:max-epochs"],
        logger=build_pl_logger(config),
        log_every_n_steps=config["output:logging:log-interval"],
        # run fewer batches per epoch (helpful when debugging)
        limit_train_batches=config["model:limit-batches:training"],
        limit_val_batches=config["model:limit-batches:validation"],
        # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#fast-dev-run
        # fast_dev_run=config["output:logging:fast-dev-run"],
    )

    trainer.fit(model, datamodule=dmod)

    LOGGER.debug("---- DONE. ----")


def main() -> None:
    """Entry point for training."""
    args = get_args()
    config = YAMLConfig(args.config)
    train(config)
