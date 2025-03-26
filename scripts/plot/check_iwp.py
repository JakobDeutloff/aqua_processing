# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from src.grid_helpers import merge_grid
from src.read_data import read_cloudsat
import pandas as pd

# %% load data
runs = ["jed0011", "jed0022", "jed0033"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
dir_name = {"jed0011": "icon-mpim", "jed0022": "icon-mpim-4K", "jed0033": "icon-mpim-2K"}
datasets = {}

# %%
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample.nc"
    )

# %% read cloudsat
cloudsat = read_cloudsat("2009")

# %% calculate iwp hist
bins = np.logspace(-4, np.log10(40), 70)
histograms = {}
for run in runs:
    iwp = datasets[run]["clivi"] + datasets[run]["qsvi"] + datasets[run]["qgvi"]
    histograms[run], edges = np.histogram(iwp, bins=bins, density=False)
    histograms[run] = histograms[run] / len(iwp)

histograms["cloudsat"], _ = np.histogram(cloudsat["ice_water_path"]/1e3, bins=bins, density=False)
histograms["cloudsat"] = histograms["cloudsat"] / len(cloudsat)

# %% plot iwp hists
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
colors = {"jed0011": "k", "jed0022": "red", "jed0033": "orange"}
ax.stairs(histograms["cloudsat"], edges, label="CloudSat", color="k", linewidth=4, alpha=0.5)
for run in runs:
    ax.stairs(histograms[run], edges, label=exp_name[run], color=colors[run])

ax.legend()
ax.set_xscale("log")
ax.set_ylabel("P(IWP)")
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.spines[["top", "right"]].set_visible(False)
#fig.savefig("plots/iwp_hist_rand.png", dpi=300)

# %% plot diff to control 
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
for run in ["jed0022", "jed0033"]:
    diff = histograms[run] - histograms["jed0011"]
    ax.stairs(diff, edges, label=exp_name[run], color=colors[run])

ax.legend()
ax.set_xscale("log")
ax.set_ylabel("P(IWP)")
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.spines[["top", "right"]].set_visible(False)
#fig.savefig("plots/iwp_hist_rand_diff.png", dpi=300)

# %%
ds_2D = xr.open_mfdataset("/work/bm1183/m301049/icon-mpim/experiments/jed0011/jed0011_atm_2d_19*.nc", chunks={}).pipe(merge_grid)

# %% plot iwp distribution in tropics for one day
day = ds_2D.time.sel(time='1979-07-29')
ds_2D_day = ds_2D.sel(time=day)[['clivi', 'qsvi', 'qgvi']].where((ds_2D.clat < 30) & (ds_2D.clat > -30), drop=True)
iwp_day = (ds_2D_day['clivi'] + ds_2D_day['qsvi'] + ds_2D_day['qgvi']).load()

# %% plot iwp distribution
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
cmap = plt.get_cmap('viridis')
norm = plt.Normalize(vmin=0, vmax=24)
for time in day:
    hist, edges = np.histogram(iwp_day.sel(time=time), bins=bins, density=False)
    hist = hist / len(iwp_day.sel(time=time))
    color = cmap(norm(time.dt.hour.values))
    ax.stairs(hist, edges, color=color)

ax.set_xscale("log")
fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, label="Time of day")
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylabel("P(IWP)")
ax.set_xlabel("IWP / kg m$^{-2}$")
fig.savefig("plots/iwp_hist_daily.png", dpi=300)

# %% plot iwp distribution for all days at one time 
times = pd.date_range(start='1979-07-01 12:00', end='1979-07-29 12:00', freq='D')
ds_2D_tod = ds_2D.sel(time=times).where((ds_2D.clat < 30) & (ds_2D.clat > -30), drop=True)
iwp_tod = (ds_2D_tod['clivi'] + ds_2D_tod['qsvi'] + ds_2D_tod['qgvi']).load()

# %% plot iwp distribution
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
cmap = plt.get_cmap('viridis')
norm = plt.Normalize(vmin=1, vmax=29)
for time in times:
    hist, edges = np.histogram(iwp_tod.sel(time=time), bins=bins, density=False)
    hist = hist / len(iwp_tod.sel(time=time))
    color = cmap(norm(time.day))
    ax.stairs(hist, edges, color=color)

ax.set_xscale("log")
fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, label="Day of month")
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylabel("P(IWP)")
ax.set_xlabel("IWP / kg m$^{-2}$")
fig.savefig("plots/iwp_hist_month.png", dpi=300)


# %%
