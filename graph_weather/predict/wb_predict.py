from typing import Optional, List
import argparse
import datetime as dt
import os

from dask.distributed import LocalCluster
from einops import rearrange
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import xarray as xr

from graph_weather.utils.dask_utils import init_dask_cluster
from graph_weather.utils.config import YAMLConfig
from graph_weather.data.wb_datamodule import WeatherBenchTestDataModule
from graph_weather.utils.logger import get_logger
from graph_weather.train.wb_trainer import LitGraphForecaster
from graph_weather.utils.constants import _NC_COMPRESS_LEVEL

LOGGER = get_logger(__name__)


def _reshape_predictions(predictions: torch.Tensor, config: YAMLConfig) -> torch.Tensor:
    """
    Reshapes the predictions:
    (batch_size, lat * lon, nvar * plevs, rollout) -> (batch_size, nvar, plevs, lat, lon, rollout)
    Args:
        predictions: predictions (already concatenated)
    Returns:
        Reshaped predictions (see above)
    """
    _NLAT_WB, _NLON_WB = 181, 360
    _PLEV_WB = 13

    l = len(config["input:variables:levels"]) if config["input:variables:levels"] is not None else _PLEV_WB

    assert predictions.shape[1] == _NLAT_WB * _NLON_WB, "Predictions tensor doesn't have the expected lat/lon shape!"
    return rearrange(
        predictions,
        "b (h w) (v l) r -> b v l h w r",
        h=_NLAT_WB,
        w=_NLON_WB,
        v=len(config["input:variables:names"]),
        l=l,
        r=config["model:rollout"],
    )


def store_predictions(
    predictions: torch.Tensor,
    sample_idx: torch.Tensor,
    config: YAMLConfig,
) -> None:
    """
    Stores the model predictions into a netCDF file.
    Args:
        predictions: predictions tensor, shape == (batch_size, nvar, plev, lat, lon, rollout)
        sample_idx: sample indices (used to match samples to valid time points in the reference predict dataset)
        config: job configuration
    TODO: add support for parallel writes with Dask (?)
    """
    # create a new xarray Dataset with similar coordinates as ds_test, plus the rollout
    ds_pred_gnn = xr.Dataset()

    with xr.open_dataset(config["input:variables:prediction:filename"]) as ds_predict:
        for var_idx, varname in enumerate(ds_predict.data_vars):
            LOGGER.debug(
                "varname: %s -- min: %.3f -- max: %.3f",
                varname,
                predictions[:, var_idx, ...].min(),
                predictions[:, var_idx, ...].max(),
            )
            plevs: xr.DataArray = (
                ds_predict.coords["level"].isel(level=config["input:variables:levels"])
                if config["input:variables:levels"] is not None
                else ds_predict.coords["level"]
            )
            ds_pred_gnn[varname] = xr.DataArray(
                data=predictions[:, var_idx, ...].squeeze().numpy(),
                dims=["time", "level", "latitude", "longitude", "rollout"],
                coords=dict(
                    latitude=ds_predict.coords["latitude"],
                    longitude=ds_predict.coords["longitude"],
                    level=plevs,
                    time=ds_predict.coords["time"].isel(time=sample_idx[0, :]),
                    rollout=np.arange(0, config["model:rollout"], dtype=np.int32),
                ),
                attrs=ds_predict[varname].attrs,
            )
            ds_pred_gnn[varname] = ds_pred_gnn[varname].sortby("time", ascending=True)

    comp = dict(zlib=True, complevel=_NC_COMPRESS_LEVEL)
    encoding = {var: comp for var in ds_pred_gnn.data_vars}
    ds_pred_gnn.to_netcdf("ds_pred_gnn.nc", encoding=encoding)


def backtransform_predictions(predictions: torch.Tensor, means: xr.Dataset, sds: xr.Dataset, config: YAMLConfig) -> torch.Tensor:
    """
    Transforms the model predictions back into the original data domain.
    ATM this entails a simple (Gaussian) re-scaling: predictions <- predictions * sds + means.
    Args:
        predictions: predictions tensor, shape == (batch_size, lat*lon, nvar*plev, rollout)
        means, sds: summary statistics calculated from the training dataset
        config: job configuration
    Returns:
        Back-transformed predictions, shape == (batch_size, nvar, plev, lat, lon, rollout)
    """
    predictions = _reshape_predictions(predictions, config)
    for ivar, varname in enumerate(means.data_vars):
        LOGGER.debug("Varname: %s, Mean: %.3f, Std: %.3f", varname, means[varname].values, sds[varname].values)
        predictions[:, ivar, ...] = predictions[:, ivar, ...] * sds[varname].values + means[varname].values
    return predictions


def predict(config: YAMLConfig, checkpoint_relpath: str) -> None:
    """
    Predict entry point.
    Args:
        config: job configuration
        checkpoint_relpath: path to the model checkpoint that you want to restore
                            should be relative to your config["output:basedir"]/config["output:checkpoints:ckpt-dir"]
    """
    # create data module (data loaders and data sets)
    dmod = WeatherBenchTestDataModule(config)

    # number of variables (features)
    num_features = dmod.ds_test.nlev * dmod.ds_test.nvar
    LOGGER.debug("Number of variables: %d", num_features)
    LOGGER.debug("Number of auxiliary (time-independent) variables: %d", dmod.const_data.nconst)

    model = LitGraphForecaster(
        lat_lons=dmod.const_data.latlons,
        feature_dim=num_features,
        aux_dim=dmod.const_data.nconst,
        hidden_dim=config["model:hidden-dim"],
        num_blocks=config["model:num-blocks"],
        lr=config["model:learn-rate"],
        rollout=config["model:rollout"],
    )

    # TODO: restore model from checkpoint
    checkpoint_filepath = os.path.join(config["output:basedir"], config["output:checkpoints:ckpt-dir"], checkpoint_relpath)
    model = LitGraphForecaster.load_from_checkpoint(checkpoint_filepath)

    # init logger
    if config["model:wandb:enabled"]:
        # use weights-and-biases
        logger = WandbLogger(
            project="GNN-WB",
            save_dir=config["output:logging:log-dir"],
        )
    elif config["model:tensorboard:enabled"]:
        # use tensorboard
        logger = TensorBoardLogger(config["output:logging:log-dir"])
    else:
        logger = False

    # fast_dev_run -> runs a single batch
    trainer = pl.Trainer(
        accelerator="gpu",
        detect_anomaly=config["model:debug:anomaly-detection"],
        # devices=config["model:num-gpus"],
        devices=1,  # run on a single GPU... for now
        precision=config["model:precision"],
        logger=logger,
        log_every_n_steps=config["output:logging:log-interval"],
        # run fewer batches per epoch (helpful when debugging)
        limit_test_batches=config["model:limit-batches:test"],
        limit_predict_batches=config["model:limit-batches:predict"],
        # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#fast-dev-run
        # fast_dev_run=config["output:logging:fast-dev-run"],
        max_epochs=-1,
    )

    # run a test loop (calculates test_wmse, returns nothing)
    trainer.test(model, datamodule=dmod)

    # run a predict loop on a different dataset - this doesn't calculate the WMSE but returns the predictions and sample indices
    predict_output = trainer.predict(model, datamodule=dmod, return_predictions=True)

    predictions_, sample_idx_ = zip(*predict_output)

    predictions: torch.Tensor = torch.cat(predictions_, dim=0).float()
    sample_idx = np.concatenate(sample_idx_, axis=-1)  # shape == (rollout + 1, num_pred_samples)

    LOGGER.debug("Backtransforming predictions to the original data space ...")
    predictions = backtransform_predictions(predictions, dmod.ds_test.mean, dmod.ds_test.sd, config)

    # save data along with observations
    LOGGER.debug("Storing model predictions to a netCDF ...")
    store_predictions(predictions, sample_idx, config)

    LOGGER.debug("---- DONE. ----")


def get_args() -> argparse.Namespace:
    """Returns a namespace containing the command line arguments"""
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("--config", required=True, help="Model configuration file (YAML)")
    required_args.add_argument(
        "--checkpoint", required=True, help="Path to the model checkpoint file (located under output-basedir/chkpt-dir)."
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for inference."""
    args = get_args()
    config = YAMLConfig(args.config)
    predict(config, args.checkpoint)
