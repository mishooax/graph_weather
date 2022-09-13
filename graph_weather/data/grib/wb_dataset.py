# Dataloader for WeatherBench (netCDF)
from typing import List, Optional, Tuple
import os

import numpy as np
# import pandas as pd

from torch.utils.data import Dataset
from graph_weather.utils.logger import get_logger

LOGGER = get_logger(__name__)


class WeatherBenchGRIBDataset(Dataset):
    """
    Iterable dataset for WeatherBench data.
    """

    # default number of pressure levels for the WeatherBench dataset
    _WB_PLEVS = 13

    def __init__(
        self,
        fdir: str,
        var_names: List[str],
        var_mean: np.ndarray,
        var_sd: np.ndarray,
        plevs: Optional[List[int]] = None,
        lead_time: int = 6,
        rollout: int = 1,
    ) -> None:
        """Initialize the internal state of the dataset."""
        super().__init__()

        self.lead_time = lead_time
        assert self.lead_time > 0 and self.lead_time % 6 == 0, "Lead time must be multiple of 6 hours"
        self.lead_step = lead_time // 6

        self.vars = var_names
        self.nvar = len(self.vars)
        self.rollout = rollout

        # pressure levels
        self.plevs = plevs
        self.nlev = len(self.plevs) if self.plevs is not None else self._WB_PLEVS

        # grib-specific stuff
        self.grib_fields_per_sample = self.nvar * self.nlev

        self.mean = var_mean
        self.sd = var_sd

        # dates = "1979-01-01/to/2015-12-31"
        # dates = list(pd.date_range("1979-01-01", "2015-12-31", freq="6H"))  # this has to be a list of strings
        # [x.strftime("%Y%m%d") for x in pd.date_range(**v["alldates"])]
        import climetlab as cml
        self.ds = cml.load_source("directory", fdir).sel(date=None, time=None, param=var_names, level=plevs)
        LOGGER.debug("Hello I am process %d with a dataset id %d", os.getpid(), id(self.ds))

    def _transform(self, data: np.ndarray, varidx: int) -> np.ndarray:
        return (data - self.mean[varidx]) / self.sd[varidx]

    def __len__(self) -> int:
        return int(len(self.ds) // self.grib_fields_per_sample)

    def _get_sample(self, i: int) -> np.ndarray:
        # for offset in range(self.grib_fields_per_sample):
        #     print(f"offset: {offset} -- field: {self.ds[i * self.grib_fields_per_sample + offset]}")
        #     print(f"converting to numpy, shape: %s", self.ds[i * self.grib_fields_per_sample + offset].to_numpy().astype(np.float32).shape)

        return np.stack(
            [
                self._transform(
                    self.ds[i * self.grib_fields_per_sample + offset].to_numpy().astype(np.float32), offset // self.nlev
                )
                for offset in range(self.grib_fields_per_sample)
            ],
            axis=0,
        )

    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        batch: List[np.ndarray] = []

        for r in range(self.rollout + 1):
            X = self._get_sample(i + r * self.lead_step)
            batch.append(X)

        return tuple(batch) + (i,)

    def __repr__(self) -> str:
        return f"""
            {super().__repr__()}
            Filenames: {str(self.fnames)}
            Varnames: {str(self.vars)}
            Plevs: {str(self.plevs)}
            Lead time: {self.lead_time}
        """
