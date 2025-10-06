# %%
import xarray as xr
import numpy as np
from src.grid_helpers import merge_grid, fix_time
from dask.diagnostics import ProgressBar

# %% load data
ds = xr.open_dataset(
    "/work/bm1183/m301049/icon_hcap_data/plus2K/production/latlon/atm2d_latlon.nc"
)
ds_3d_1 = (
    xr.open_mfdataset(
        "/work/bm1183/m301049/icon-mpim-2K/experiments/jed0333/jed0333_atm_3d_*.nc"
    )
    .pipe(fix_time)
    .pipe(merge_grid)[["hus", "va", "ta", "rho"]]
)
ds_3d_2 = (
    xr.open_mfdataset(
        "/work/bm1183/m301049/icon-mpim-2K/experiments/jed0033/jed0033_atm_3d_*.nc"
    )
    .pipe(fix_time)
    .pipe(merge_grid)[["hus", "va", "ta", "rho"]]
)
ds_3d = xr.concat([ds_3d_1, ds_3d_2], dim="time")
ds_3d = ds_3d.chunk({"time": 1, "ncells": -1})

# %%
import pandas as pd

with ProgressBar():
    ds_north = (
        ds_3d.where((ds_3d["clat"] < 15) & (ds_3d["clat"] > 14.9))
        .groupby_bins(ds_3d["clon"], bins=np.arange(-180, 180.1, 0.1))
        .mean()
    )
    # Convert clon_bins from Interval to float (left edge)
    ds_north = ds_north.assign_coords(
        clon_bins=[
            iv.left if isinstance(iv, pd.Interval) else float(iv)
            for iv in ds_north.clon_bins.values
        ]
    )
    ds_north.to_netcdf(
        "/work/bm1183/m301049/icon_hcap_data/plus2K/production/latlon/zonal_north_15.nc",
        mode="w",
    )

    ds_south = (
        ds_3d.where((ds_3d["clat"] > -15) & (ds_3d["clat"] < -14.9))
        .groupby_bins(ds_3d["clon"], bins=np.arange(-180, 180.1, 0.1))
        .mean()
    )
    ds_south = ds_south.assign_coords(
        clon_bins=[
            iv.left if isinstance(iv, pd.Interval) else float(iv)
            for iv in ds_south.clon_bins.values
        ]
    )
    ds_south.to_netcdf(
        "/work/bm1183/m301049/icon_hcap_data/plus2K/production/latlon/zonal_south_15.nc",
        mode="w",
    )
