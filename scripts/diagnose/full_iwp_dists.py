# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from src.grid_helpers import merge_grid, fix_time
import pickle as pkl

# %% load data
runs = ["jed0011", "jed0022", "jed0033"]
followups = {"jed0011": "jed0111", "jed0022": "jed0222", "jed0033": "jed0333"}
configs = {"jed0011": "icon-mpim", "jed0022": "icon-mpim-4K", "jed0033": "icon-mpim-2K"}
colors = {
    "jed0011": "k",
    "jed0022": "r",
    "jed0033": "orange",
}
datasets = {}
for run in runs:
    ds_first_month = (
        xr.open_mfdataset(
            f"/work/bm1183/m301049/{configs[run]}/experiments/{run}/{run}_atm_2d_19*.nc"
        )
        .pipe(merge_grid)
        .pipe(fix_time)[["clivi", "qsvi", "qgvi"]]
    )
    ds_first_month = ds_first_month.sel(
        time=(ds_first_month.time.dt.minute == 0) & (ds_first_month.time.dt.hour == 0)
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
        & (ds_last_two_months.time.dt.hour == 0)
    )
    datasets[run] = xr.concat([ds_first_month, ds_last_two_months], dim="time").astype(float)

# %% calculate IWP
for run in runs:
    print(run)
    datasets[run] = (
        datasets[run]
        .where((datasets[run].clat < 20) & (datasets[run].clat > -20), drop=True)
        .load()
    )
    datasets[run] = datasets[run].assign(
        {"iwp": datasets[run]["clivi"] + datasets[run]["qsvi"] + datasets[run]["qgvi"]}
    )


# %% calculate IWP histograms
hists = {}
iwp_bins = np.logspace(-4, np.log10(40), 51)
for run in runs:
    hist, edges = np.histogram(datasets[run]["iwp"], bins=iwp_bins)
    hists[run] = hist / datasets[run]["iwp"].size


# %% save iwp distributions 
with open('/work/bm1183/m301049/icon_hcap_data/iwp_dists.pkl', 'wb') as f:
    pkl.dump(hists, f)

# %% calculate local time 
for run in runs: 
    datasets[run] = datasets[run].where(datasets[run]["iwp"] > 1, drop=True)
    datasets[run] = datasets[run].assign(
        time_local=lambda d: d.time.dt.hour + (d.clon / 15)
    )
    datasets[run]["time_local"] = (
        datasets[run]["time_local"]
        .where(datasets[run]["time_local"] < 24, datasets[run]["time_local"] - 24)
        .where(datasets[run]["time_local"] > 0, datasets[run]["time_local"] + 24)
    )
    datasets[run]["time_local"].attrs = {"units": "h", "long_name": "Local time"}
# %% calculate P(I>1)
bins = np.arange(0, 25, 1)
hists_deep = {}
for run in runs:
    hist, edges = np.histogram(
        datasets[run]["time_local"].where(datasets[run]["iwp"] > 1e0),
        bins=bins,
        density=True,
    )
    hists_deep[run] = hist

# %% quick plot

fig, ax = plt.subplots(figsize=(8, 6))
for run in runs:
    ax.stairs(
        hists_deep[run],
        edges,
        color=colors[run],
        label=f"{run}",
    )
# %%
