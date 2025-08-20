# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from src.grid_helpers import merge_grid, fix_time
import pickle as pkl
import sys
from dask.diagnostics import ProgressBar

# %% load data
run = sys.argv[1]
followups = {"jed0011": "jed0111", "jed0022": "jed0222", "jed0033": "jed0333"}
configs = {"jed0011": "icon-mpim", "jed0022": "icon-mpim-4K", "jed0033": "icon-mpim-2K"}
names = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}

ds_first_month = (
    xr.open_mfdataset(
        f"/work/bm1183/m301049/{configs[run]}/experiments/{run}/{run}_atm_2d_19*.nc"
    )
    .pipe(merge_grid)
    .pipe(fix_time)[["clivi", "qsvi", "qgvi"]]
)
ds_first_month = ds_first_month.sel(
    time=(ds_first_month.time.dt.minute == 0)
)
ds_last_two_months = (
    xr.open_mfdataset(
        f"/work/bm1183/m301049/{configs[run]}/experiments/{followups[run]}/{followups[run]}_atm_2d_19*.nc"
    )
    .pipe(merge_grid)
    .pipe(fix_time)[["clivi", "qsvi", "qgvi"]]
)
ds_last_two_months = ds_last_two_months.sel(
    time=(ds_last_two_months.time.dt.minute == 0)
)
dataset = xr.concat([ds_first_month, ds_last_two_months], dim="time").astype(float)

# %% calculate IWP
dataset = dataset.where((dataset.clat < 20) & (dataset.clat > -20), drop=True)
iwp = (dataset["clivi"] + dataset["qsvi"] + dataset["qgvi"])

# %% save iwp 
with ProgressBar():
    iwp.to_netcdf(f'/work/bm1183/m301049/icon_hcap_data/{names[run]}/production/{run}_iwp.nc')