# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# %% load data
runs = ["jed0011", "jed0033", "jed0022"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
colors = {"jed0011": "k", "jed0022": "r", "jed0033": "orange"}
datasets = {}
hrs = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed_20.nc"
    )

vgrid = (
    xr.open_dataset(
        "/work/bm1183/m301049/icon-mpim/experiments/jed0001/atm_vgrid_angel.nc"
    )
    .mean("ncells")
    .rename({"height": "height_2", "height_2": "height"})
)
# %% plot precipitation binned by IWP
iwp_bins = np.logspace(-4, 1, 51)
iwp_points = (iwp_bins[:-1] + iwp_bins[1:]) / 2
fig, ax = plt.subplots()

mean_control = (
    datasets["jed0011"]["pr"]
    .groupby_bins(datasets["jed0011"]["iwp"], bins=iwp_bins)
    .mean()
)
for run in ["jed0033", "jed0022"]:
    ax.plot(
        iwp_points,
        datasets[run]["pr"].groupby_bins(datasets[run]["iwp"], bins=iwp_bins).mean() - mean_control,
        label=exp_name[run],
        color=colors[run],
    )
ax.set_xscale("log")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("$I$ / kg m$^{-2}$")
ax.spines[["top", "right"]].set_visible(False)
ax.legend()
ax.set_ylabel("Precipitation Difference / mm s$^{-1}$")
ax.set_xlim(1e-4, 10)
fig.savefig('plots/iwp_drivers/pr_diff.png', dpi=300, bbox_inches="tight")
# %%
