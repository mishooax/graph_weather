from typing import List, Tuple
import unittest
import logging

import numpy as np
import xarray as xr
import dask.array as da
from einops import rearrange

import torch
from torch.utils.data import DataLoader

from graph_weather.data.wb_dataset import WeatherBenchDataset, worker_init_func

# dummy xarray data
np.random.seed(0)

NTIME = 100
NLON, NLAT = 2, 2
NLEV = 13
PLEVELS = [0, 1, 2]
NVAR = 1
BATCH_SIZE = 2
BATCH_CHUNK_SIZE = 3
LEAD_TIME = 18
NUM_WORKERS = 2

lon = [[-9.83, -9.32], [-9.79, -9.23]]
lat = [[42.25, 42.21], [42.63, 42.59]]
time = np.array(range(NTIME))
plev = np.array(range(NLEV))

# init t and z values with the time index value (should make it easier to identify later)
t = np.zeros((NTIME, NLEV, NLAT, NLON), dtype=np.int32)
for i in range(NTIME):
    t[i, ...] = np.ones((NLEV, NLAT, NLON)) * i

DATA = xr.Dataset(
    data_vars=dict(
        t=(["time", "level", "x", "y"], t),
    ),
    coords=dict(
        lon=(["x", "y"], lon),
        lat=(["x", "y"], lat),
        time=time,
        level=plev,
    ),
    attrs=dict(description="Dummy data."),
)

t_dummy_mean = np.zeros((NTIME, NLEV, NLAT, NLON), dtype=np.int32)
t_dummy_sd = np.ones((NTIME, NLEV, NLAT, NLON), dtype=np.int32)

DATA_MEANS = xr.Dataset(
    data_vars=dict(
        t=(["time", "level", "x", "y"], t_dummy_mean),
    ),
    coords=dict(
        lon=(["x", "y"], lon),
        lat=(["x", "y"], lat),
        time=time,
        level=plev,
    ),
    attrs=dict(description="Dummy data mean."),
)

DATA_SDS = xr.Dataset(
    data_vars=dict(
        t=(["time", "level", "x", "y"], t_dummy_sd),
    ),
    coords=dict(
        lon=(["x", "y"], lon),
        lat=(["x", "y"], lat),
        time=time,
        level=plev,
    ),
    attrs=dict(description="Dummy data sd."),
)


def test_batch_collator(batch_data: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collation function. It collates several batch chunks into a "full" batch.
    Args:
        data: batch data, [(X_chunk0, Y_chunk0), (X_chunk1, Y_chunk1), ...]
              with X_chunk0.shape == (batch_chunk_size, nvars, nlevels, lat, lon)
    """
    X, Y = list(zip(*batch_data))
    return torch.as_tensor(
        # reshape to (bs, (lat*lon), (nvar * nlev))
        rearrange(da.concatenate([x for x in X], axis=0).compute(), "b v l h w -> b (h w) (v l)"),
        dtype=torch.int32,
    ), torch.as_tensor(
        rearrange(da.concatenate([y for y in Y], axis=0).compute(), "b v l h w -> b (h w) (v l)"),
        dtype=torch.int32,
    )


def read_dummy_data(fnames: List[str]) -> xr.Dataset:
    del fnames  # not used
    return DATA


class DataloaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.yyyymmdd = "20220812"
        cls.hh = "00"
        logging.disable(logging.CRITICAL)  # disable logging

    @classmethod
    def tearDownClass(cls):
        pass

    def test_dataloader(self):
        ds_test = WeatherBenchDataset(
            fnames=[],  # the data is already in memory
            read_wb_data_func=read_dummy_data,
            var_names=["t"],
            var_mean=DATA_MEANS,
            var_sd=DATA_SDS,
            plevs=PLEVELS,
            lead_time=LEAD_TIME,
            batch_chunk_size=BATCH_CHUNK_SIZE,
        )

        dl_test = DataLoader(
            ds_test,
            # we're putting together one full batch from this many batch-chunks
            # this means the "real" batch size == config["model:dataloader:batch-size"] * config["model:dataloader:batch-chunk-size"]
            batch_size=BATCH_SIZE,
            # number of worker processes
            num_workers=NUM_WORKERS,
            # use of pinned memory can speed up CPU-to-GPU data transfers
            # see https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning
            pin_memory=True,
            # custom collator (see above)
            collate_fn=test_batch_collator,
            # worker initializer
            worker_init_fn=worker_init_func,
            # prefetch batches (default prefetch_factor == 2)
            prefetch_factor=2,
            # drop last incomplete batch (makes it easier to test the resulting tensor shapes)
            drop_last=True,
        )

        for batch in dl_test:
            # check batch shapes
            print(batch[0].shape, batch[1].shape)
            self.assertEqual(batch[0].shape, batch[1].shape)
            self.assertEqual(batch[0].shape, (BATCH_SIZE * BATCH_CHUNK_SIZE, NLAT * NLON, len(PLEVELS) * NVAR))
            X, Y = batch
            # the entries of Y - X should all be equal to the input-vs-target index offset value (i.e. lead_time // 6)
            self.assertTrue(((Y - X) == (LEAD_TIME // 6)).all())
