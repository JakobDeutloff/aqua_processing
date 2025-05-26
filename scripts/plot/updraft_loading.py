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

# %% calculate IWP
for run in runs:
    print(run)
    datasets[run] = (
        datasets[run]
        .where((datasets[run].clat < 20) & (datasets[run].clat > -20), drop=True)
        .load()
    )
    datasets[run] = datasets[run].assign(
        {"iwp": datasets[run]["clivi"] + datasets[run]["qsvi"] + datasets[run]["qgvi"], 
         "lwp": datasets[run]["cllvi"] + datasets[run]["qrvi"]}
    )


# %% calculate IWP histograms
hists = {}
iwp_bins = np.logspace(-4, np.log10(40), 51)
for run in runs:
    hist, edges = np.histogram(datasets[run]["iwp"], bins=iwp_bins)
    hists[run] = hist / datasets[run]["iwp"].size

# %% plot histograms
fig, ax = plt.subplots(figsize=(8, 6))
for run in runs:
    ax.stairs(hists[run], edges, label=run, color=colors[run])

ax.set_xscale("log")

# %% plot change relative to control 
fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

for run in runs[1:]:
    axes[0].stairs(
        (hists[run] - hists[runs[0]]) / hists[runs[0]],
        edges,
        label=run,
        color=colors[run],
    )
    axes[1].stairs(
        (hists[run] - hists[runs[0]]),
        edges,
        label=run,
        color=colors[run],
    )

axes[0].set_xscale("log")

# %% calculate IWP and LWP percentiles 
percentiles = np.arange(95, 100, 0.005)
iwp_percentiles = {}
lwp_percentiles = {}
for run in runs:
    print(run)
    iwp_percentiles[run] = np.percentile(
        datasets[run]["iwp"].values, percentiles
    )
    iwp_percentiles[run] = xr.DataArray(
        iwp_percentiles[run], coords=[percentiles], dims=["percentiles"]
    )
    iwp_percentiles[run].attrs["units"] = "kg/m^2"
    iwp_percentiles[run].attrs["long_name"] = "IWP percentiles"
    lwp_percentiles[run] = np.percentile(
        datasets[run]["lwp"].values, percentiles
    )
    lwp_percentiles[run] = xr.DataArray(
        lwp_percentiles[run], coords=[percentiles], dims=["percentiles"]
    )
    lwp_percentiles[run].attrs["units"] = "kg/m^2"
    lwp_percentiles[run].attrs["long_name"] = "LWP percentiles"


# %% plot iwp and lwp percentiles
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
for run in runs:
    axes[0, 0].plot(
        iwp_percentiles[run].percentiles,
        iwp_percentiles[run],
        label=run,
        color=colors[run],
    )
    axes[0, 1].plot(
        lwp_percentiles[run].percentiles,
        lwp_percentiles[run],
        label=run,
        color=colors[run],
    )

for run in runs[1:]:
    axes[1, 0].plot(
        iwp_percentiles[run].percentiles,
        (iwp_percentiles[run] - iwp_percentiles[runs[0]]) * 100 / iwp_percentiles[runs[0]] / t_deltas[run],
        label=run,
        color=colors[run],
    )
    axes[1, 1].plot(
        lwp_percentiles[run].percentiles,
        (lwp_percentiles[run] - lwp_percentiles[runs[0]]) * 100 / lwp_percentiles[runs[0]] / t_deltas[run],
        label=run,
        color=colors[run],
    )

for ax in axes[0, :]:
    ax.set_yscale('log')
    ax.spines[["top", "right"]].set_visible(False)

for ax in axes[1, :]:
    ax.axhline(0, color="black", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

axes[0, 0].set_ylabel("IWP / kg m$^{-2}$")
axes[0, 1].set_ylabel("LWP / kg m$^{-2}$")
axes[1, 0].set_ylabel('% / K')
axes[1, 1].set_ylabel('% / K')
axes[1, 0].set_xlabel("IWP Percentiles / %")
axes[1, 1].set_xlabel("LWP Percentiles / %")

fig.tight_layout()
fig.savefig('plots/publication/iwp_lwp_percentiles.png', dpi=300, bbox_inches='tight')
