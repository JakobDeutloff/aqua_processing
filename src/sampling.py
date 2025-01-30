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
    time = pd.date_range(
        start=ds_first.isel(time=1).time.values,
        end=ds_last.isel(time=-1).time.values,
        freq="6h",
    )
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
        print("Processing " + file)
        ds = xr.open_dataset(file, chunks={})
        time_subset = time[(time >= ds.time.values[0]) & (time <= ds.time.values[-1])]
        ds_coarse = ds.sel(time=time_subset)
        with ProgressBar():
            ds_coarse.to_netcdf(file[:-3] + "_coarse.nc")
            os.remove(file)
            os.rename(file[:-3] + "_coarse.nc", file)


def subsample_file(file, exp_name):
    """
    Get random sample of data from file
    """

    ds = xr.open_dataset(
        file,
        chunks={},
    )
    random_coords = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name}/production/random_sample/random_coords.nc"
    )

    # get overlap
    valid_time_mask = random_coords.time.isin(ds.time)
    valid_coords = random_coords.where(valid_time_mask, drop=True)

    # select data
    ds_random = ds.sel(
        ncells=valid_coords.ncells.astype(int), time=valid_coords.time
    ).assign_coords(time=valid_coords.time, ncells=valid_coords.ncells)

    # save in random sample folder
    filename = file.split("/")[-1]
    ds_random.to_netcdf(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name}/production/random_sample/{filename}"
    )
