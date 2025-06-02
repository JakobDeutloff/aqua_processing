# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from src.read_data import load_daily_2d_data

# %% load data
runs = ["jed0011", "jed0022", "jed0033"]
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
datasets = load_daily_2d_data(
    ["clivi", "qsvi", "qgvi", "cllvi", "qrvi", "pr"], "day", load=False
)

# %% calculate IWP
for run in runs:
    print(run)
    datasets[run] = datasets[run].isel(time=slice(-30, None))
    datasets[run] = datasets[run].assign(
        {
            "iwp": datasets[run]["clivi"]
            + datasets[run]["qsvi"]
            + datasets[run]["qgvi"],
            "lwp": datasets[run]["cllvi"] + datasets[run]["qrvi"],
        }
    )
    datasets[run] = datasets[run][["iwp", "lwp", "pr"]].load()


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
    iwp_percentiles[run] = np.percentile(datasets[run]["iwp"].values, percentiles)
    iwp_percentiles[run] = xr.DataArray(
        iwp_percentiles[run], coords=[percentiles], dims=["percentiles"]
    )
    iwp_percentiles[run].attrs["units"] = "kg/m^2"
    iwp_percentiles[run].attrs["long_name"] = "IWP percentiles"
    lwp_percentiles[run] = np.percentile(datasets[run]["lwp"].values, percentiles)
    lwp_percentiles[run] = xr.DataArray(
        lwp_percentiles[run], coords=[percentiles], dims=["percentiles"]
    )
    lwp_percentiles[run].attrs["units"] = "kg/m^2"
    lwp_percentiles[run].attrs["long_name"] = "LWP percentiles"

# %% calculate precip efficiency
pr = {}
for run in runs:
    mask = datasets[run]["iwp"] > 1
    pr[run] = datasets[run]["pr"].where(mask).mean().values

pr_increase = {}
for run in runs[1:]:
    pr_increase[run] = (pr[run] - pr[runs[0]])  * 100 / pr[runs[0]] / t_deltas[run]


# %% 

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
        (iwp_percentiles[run] - iwp_percentiles[runs[0]])
        * 100
        / iwp_percentiles[runs[0]]
        / t_deltas[run],
        label=run,
        color=colors[run],
    )
    axes[1, 1].plot(
        lwp_percentiles[run].percentiles,
        (lwp_percentiles[run] - lwp_percentiles[runs[0]])
        * 100
        / lwp_percentiles[runs[0]]
        / t_deltas[run],
        label=run,
        color=colors[run],
    )

for ax in axes[0, :]:
    ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)

for ax in axes[1, :]:
    ax.axhline(0, color="black", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

axes[0, 0].set_ylabel("IWP / kg m$^{-2}$")
axes[0, 1].set_ylabel("LWP / kg m$^{-2}$")
axes[1, 0].set_ylabel("% / K")
axes[1, 1].set_ylabel("% / K")
axes[1, 0].set_xlabel("IWP Percentiles / %")
axes[1, 1].set_xlabel("LWP Percentiles / %")

fig.tight_layout()
fig.savefig("plots/publication/iwp_lwp_percentiles.png", dpi=300, bbox_inches="tight")
