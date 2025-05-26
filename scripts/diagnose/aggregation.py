# %%
import cloudmetrics as cm 
import xarray as xr
import matplotlib.pyplot as plt
from src.grid_helpers import merge_grid, fix_time
import sys, os
sys.path.append(f"{os.path.expanduser('~')}/pyicon")
import pyicon 

# %% load data
runs = ["jed0011", "jed0022", "jed0033"]
followups = {"jed0011": "jed0111", "jed0022": "jed0222", "jed0033": "jed0333"}
configs = {"jed0011": "icon-mpim", "jed0022": "icon-mpim-4K", "jed0033": "icon-mpim-2K"}
colors = {
    "jed0011": "k",
    "jed0022": "r",
    "jed0033": "orange",
}
t_deltas = {
    "jed0011": 0,
    "jed0022": 4,
    "jed0033": 2,
}
datasets = {}
for run in runs:
    ds_first_month = (
        xr.open_mfdataset(
            f"/work/bm1183/m301049/{configs[run]}/experiments/{run}/{run}_atm_2d_19*.nc"
        )
        .pipe(merge_grid)
        .pipe(fix_time)[["clivi", "qsvi", "qgvi", "qrvi", 'cllvi']]
    )
    ds_first_month = ds_first_month.sel(
        time=(ds_first_month.time.dt.minute == 0) & (ds_first_month.time.dt.hour == 0)
    )
    ds_last_two_months = (
        xr.open_mfdataset(
            f"/work/bm1183/m301049/{configs[run]}/experiments/{followups[run]}/{followups[run]}_atm_2d_19*.nc"
        )
        .pipe(merge_grid)
        .pipe(fix_time)[["clivi", "qsvi", "qgvi", "qrvi", 'cllvi']]
    )
    ds_last_two_months = ds_last_two_months.sel(
        time=(ds_last_two_months.time.dt.minute == 0)
        & (ds_last_two_months.time.dt.hour == 0)
    )
    datasets[run] = xr.concat([ds_first_month, ds_last_two_months], dim="time").astype(float)

# %% calculate iwp  
ds = datasets['jed0011'].isel(time=-1)
iwp = (ds['clivi'] + ds['qsvi'] + ds['qgvi']).load()

# %% regrid to latlon 
pyicon.interp_to_rectgrid_xr(iwp)

# %%
cm.objects.iorg()