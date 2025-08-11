# %%
import numpy as np
import xarray as xr
from src.grid_helpers import merge_grid, fix_time
import numpy as np
import sys
from tqdm import tqdm
import os

# %% load data
run = sys.argv[1]
print(f"Processing run: {run}")
followups = {"jed0011": "jed0111", "jed0022": "jed0222", "jed0033": "jed0333"}
configs = {"jed0011": "icon-mpim", "jed0022": "icon-mpim-4K", "jed0033": "icon-mpim-2K"}
names = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
datasets = {}

ds_first_month = (
    xr.open_mfdataset(
        f"/work/bm1183/m301049/{configs[run]}/experiments/{run}/{run}_atm_2d_19*.nc"
    )
    .pipe(merge_grid)
    .pipe(fix_time)[["clivi", "qsvi", "qgvi"]]
).isel(time=slice(None, None, 3))
ds_last_two_months = (
    xr.open_mfdataset(
        f"/work/bm1183/m301049/{configs[run]}/experiments/{followups[run]}/{followups[run]}_atm_2d_19*.nc"
    )
    .pipe(merge_grid)
    .pipe(fix_time)[["clivi", "qsvi", "qgvi"]]
)
datasets[run] = xr.concat([ds_first_month, ds_last_two_months], dim="time").chunk({"time": 1})

# %% calculate IWP
clats = datasets[run].clat.compute()
datasets[run] = datasets[run].where(
    (clats < 20) & (clats > -20), drop=True
)
datasets[run] = datasets[run].assign(
    {"iwp": datasets[run]["clivi"] + datasets[run]["qsvi"] + datasets[run]["qgvi"]}
)

# %% select timeslice from datasets 
time_slices = np.arange(0, datasets[run].time.size, 24)

hists = xr.DataArray(
    np.zeros((len(time_slices) - 1, 24)),
    dims=['day', "local_hour"],
    coords={
        "day": np.arange(0, len(time_slices) - 1),
        "local_hour": np.arange(0, 24)
    }
)
for i in tqdm(range(0, len(time_slices) - 1)):
    start = time_slices[i]
    end = time_slices[i + 1]
    sample = datasets[run][["iwp"]].isel(time=slice(start, end)).compute()
    #  calculate local_time
    sample = sample.assign(
        time_local=lambda d: d.time.dt.hour + (d.clon / 15)
    )
    sample["time_local"] = (
        sample["time_local"]
        .where(sample["time_local"] < 24, sample["time_local"] - 24)
        .where(sample["time_local"] > 0, sample["time_local"] + 24)
    )
    # calculate histogram
    bins = np.arange(0, 25, 1)
    hist, edges = np.histogram(
        sample["time_local"].where(sample["iwp"] > 5),
        bins=bins,
        density=False,
    )
    hists[i, :] = hist

# %% save hists
path = f"/work/bm1183/m301049/icon_hcap_data/{names[run]}/production/deep_clouds_daily_cycle_5.nc"
if os.path.exists(path):
    os.remove(path)
hists.to_netcdf(path)

