from typing import Callable, List, Tuple
import os

from einops import rearrange
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import xarray as xr

from graph_weather.data.grib.wb_dataset import WeatherBenchGRIBDataset
from graph_weather.data.wb_constants import WeatherBenchConstantFields
from graph_weather.utils.config import YAMLConfig
from graph_weather.utils.logger import get_logger

LOGGER = get_logger(__name__)


class WeatherBenchDataBatch:
    """Custom batch type for WeatherBench data."""

    def __init__(self, batch_data, const_data: np.ndarray) -> None:
        """Construct a batch object from the variable and constant data tensors."""
        zipped_batch = list(zip(*batch_data))

        batch: List[torch.Tensor] = []
        for X in zipped_batch[:-1]:
            X = torch.as_tensor(
                np.concatenate(
                    [
                        # reshape to (bs, (lat*lon), (nvar * nlev))
                        rearrange(np.stack([x for x in X], axis=0), "b vl h w -> b (h w) vl"),
                        # reshape to (bs, (lat*lon), nconst)
                        rearrange(const_data, "b h w c -> b (h w) c"),
                    ],
                    # concat along last axis (var index)
                    # final shape: (bs, (lat*lon), nvar * nlev + nconst) -> this is what the GNN expects
                    axis=-1,
                )
            )
            batch.append(X)

        self.X: Tuple[torch.Tensor] = tuple(batch)
        self.idx: torch.Tensor = torch.as_tensor(np.array(zipped_batch[-1], dtype=np.int32))

    def pin_memory(self):
        """Custom memory pinning. See https://pytorch.org/docs/stable/data.html#memory-pinning."""
        self.X = tuple(t.pin_memory() for t in self.X)
        self.idx = self.idx.pin_memory()
        return self


def _custom_collator_wrapper(const_data: np.ndarray) -> Callable:
    def custom_collator(batch_data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Custom collation function. It collates several batch chunks into a "full" batch.
        """
        return WeatherBenchDataBatch(batch_data=batch_data, const_data=const_data)

    return custom_collator


class WeatherBenchGRIBDataModule(pl.LightningDataModule):
    def __init__(self, config: YAMLConfig) -> None:
        super().__init__()
        self.config = config

        if config["input:variables:training:summary-stats:precomputed"]:
            var_means, var_sds = self._load_summary_statistics()
        else:
            raise Exception("Input statistics must be precomputed! Check your config file.")

        self.ds_train = WeatherBenchGRIBDataset(
            fdir=config["input:variables:training:basedir"],
            var_names=config["input:variables:names"],
            var_mean=var_means,
            var_sd=var_sds,
            plevs=config["input:variables:levels"],
            lead_time=config["model:lead-time"],
            rollout=config["model:rollout"],
        )

        self.ds_valid = WeatherBenchGRIBDataset(
            fdir=config["input:variables:validation:basedir"],
            var_names=config["input:variables:names"],
            var_mean=var_means,
            var_sd=var_sds,
            plevs=config["input:variables:levels"],
            lead_time=config["model:lead-time"],
            rollout=config["model:rollout"],
        )

        self.const_data = WeatherBenchConstantFields(
            const_fname=config["input:constants:filename"],
            const_names=config["input:constants:names"],
        )

    def _load_summary_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        # load pre-computed means and standard deviations
        var_names = self.config["input:variables:names"]
        var_means = xr.load_dataset(
            os.path.join(
                self.config["input:variables:training:basedir"], self.config["input:variables:training:summary-stats:means"]
            )
        )
        var_sds = xr.load_dataset(
            os.path.join(
                self.config["input:variables:training:basedir"], self.config["input:variables:training:summary-stats:std-devs"]
            )
        )
        return var_means[var_names].to_array().values, var_sds[var_names].to_array().values

    def train_dataloader(self) -> DataLoader:
        bs: int = self.config["model:dataloader:batch-size:training"]
        return DataLoader(
            self.ds_train,
            batch_size=bs,
            # shuffle=True,
            # number of worker processes
            num_workers=self.config["model:dataloader:num-workers:training"],
            # use of pinned memory can speed up CPU-to-GPU data transfers
            pin_memory=True,
            # custom collator (see above)
            collate_fn=_custom_collator_wrapper(self.const_data.get_constants(bs)),
            # prefetch_factor=4,
            # persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        bs: int = self.config["model:dataloader:batch-size:validation"]
        return DataLoader(
            self.ds_valid,
            batch_size=bs,
            num_workers=self.config["model:dataloader:num-workers:validation"],
            pin_memory=True,
            collate_fn=_custom_collator_wrapper(self.const_data.get_constants(bs)),
            # prefetch_factor=4,
            # persistent_workers=True,
        )

    def transfer_batch_to_device(self, batch: WeatherBenchDataBatch, device: torch.device, dataloader_idx: int = 0) -> None:
        del dataloader_idx  # not used
        batch.X = tuple(x.to(device) for x in batch.X)
        batch.idx = batch.idx.to(device)
        return batch
