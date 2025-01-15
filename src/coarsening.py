import xarray as xr
import pandas as pd
from dask.diagnostics import ProgressBar
import glob
import os

def get_coarse_time(first, last, path):
    """
    Build new time axis with 6h frequency
    """
    ds_first = xr.open_dataset(path + first, chunks={})
    ds_last = xr.open_dataset(path + last, chunks={})
    time = pd.date_range(start=ds_first.isel(time=1).time.values, end=ds_last.isel(time=-1).time.values, freq='6h')
    return time

def get_ds_list(path, pattern):
    """
    Get list of all datasets in path matching pattern
    """
    ds_list = glob.glob(path + pattern)
    ds_list.sort()
    return ds_list

def coarsen_ds(ds_list, time):
    """
    Coarsen datasets in ds_list to new time axis
    """
    for file in ds_list:
        print('Processing ' + file)
        ds = xr.open_dataset(file, chunks={})
        time_subset = time[(time >= ds.time.values[0]) & (time <= ds.time.values[-1])]
        ds_coarse = ds.sel(time=time_subset)
        with ProgressBar():
            ds_coarse.to_netcdf(file[:-3] + '_coarse.nc')
            os.remove(file)
            os.rename(file[:-3] + '_coarse.nc', file)

