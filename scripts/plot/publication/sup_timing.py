# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from src.read_data import load_random_datasets, load_definitions, load_vgrid
import pandas as pd

# %% load CRE data
runs, exp_name, colors, line_labels, sw_color, lw_color, net_color, linestyles = (
    load_definitions()
)
iwp_bins = np.logspace(-4, np.log10(40), 51)
iwp_points = (iwp_bins[:-1] + iwp_bins[1:]) / 2
datasets = load_random_datasets()
vgrid = load_vgrid()
ds_const_o3 = xr.open_dataset(
    "/work/bm1183/m301049/icon_hcap_data/const_o3/production/random_sample/jed2224_randsample_processed_64.nc"
)

# %% mean SW down for I>1
for run in runs:
    sw_down = datasets[run]["rsdt"].where(datasets[run]["iwp"] > 1).mean()
    print(f"{run} {sw_down.values}")

# %% plot median local time
fig, ax = plt.subplots(figsize=(8, 5))
for run in runs:
    median_time = (
        datasets[run]["time_local"]
        .groupby_bins(datasets[run]["iwp"], iwp_bins)
        .median()
    )
    ax.plot(
        iwp_points,
        median_time,
        label=exp_name[run],
        color=colors[run],
    )

ax.set_ylabel("Local Time / h")
ax.set_xlabel("$I$ / $kg m^{-2}$")
ax.set_xscale("log")


# %% bin quantities
iwp_bins = np.logspace(-4, np.log10(40), 51)
iwp_points = (iwp_bins[:-1] + iwp_bins[1:]) / 2
time_binned = {}
rad_time_binned = {}
sw_down_binned = {}
lat_binned = {}
time_std = {}
temp_binned = {}
for run in runs:
    time_binned[run] = (
        datasets[run]["time_local"].groupby_bins(datasets[run]["iwp"], iwp_bins).mean()
    )
    sw_down_binned[run] = (
        datasets[run]["rsdt"].groupby_bins(datasets[run]["iwp"], iwp_bins).mean()
    )
    lat_binned[run] = (
        np.abs(datasets[run]["clat"])
        .groupby_bins(datasets[run]["iwp"], iwp_bins)
        .mean()
    )
    rad_time_binned[run] = (
        np.abs(datasets[run]["time_local"] - 12)
        .groupby_bins(datasets[run]["iwp"], iwp_bins)
        .mean()
    )
    temp_binned[run] = (
        datasets[run]["hc_top_temperature"]
        .groupby_bins(datasets[run]["iwp"], iwp_bins)
        .mean()
    )

# %% plot mean time and SW down
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

for run in runs:
    sw_down_binned[run].sel(iwp_bins=slice(1e-4, 10)).plot(
        ax=axes[0], label=exp_name[run], color=colors[run]
    )
    rad_time_binned[run].sel(iwp_bins=slice(1e-4, 10)).plot(
        ax=axes[1], label=exp_name[run], color=colors[run]
    )
    lat_binned[run].sel(iwp_bins=slice(1e-4, 10)).plot(
        ax=axes[2], label=exp_name[run], color=colors[run]
    )

axes[0].set_ylabel("SW down / W m$^{-2}$")
axes[1].set_ylabel("Time Difference to Noon / h")
axes[2].set_ylabel("Distance to equator / deg")
axes[0].set_xscale("log")
for ax in axes:
    ax.set_xlabel("$I$ / $kg m^{-2}$")
    ax.spines[["top", "right"]].set_visible(False)

axes[1].invert_yaxis()
fig.savefig("plots/publication/S1.pdf", bbox_inches="tight")


# %%
