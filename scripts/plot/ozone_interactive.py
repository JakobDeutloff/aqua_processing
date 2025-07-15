# %%
import xarray as xr
from src.grid_helpers import merge_grid, fix_time
import matplotlib.pyplot as plt
import numpy as np
from src.read_data import load_random_datasets, load_definitions

# %%
bc_ozone = xr.open_dataset(
    "/pool/data/ICON/grids/public/mpim/0015/ozone/r0002/bc_ozone_ape.nc"
)
ozone_control = (
    xr.open_dataset(
        "/work/bm1183/m301049/icon-mpim/experiments/jed0111/jed0111_atm_3d_main_19790928T000040Z.16636549.nc"
    )
    .pipe(merge_grid)
    .pipe(fix_time)[["o3", "pfull"]]
).astype(float)
ozone_4K = (
    xr.open_dataset(
        "/work/bm1183/m301049/icon-mpim-4K/experiments/jed0222/jed0222_atm_3d_main_19791128T000040Z.16501620.nc"
    )
    .pipe(merge_grid)
    .pipe(fix_time)[["o3", "pfull"]]
).astype(float)

datasets = load_random_datasets()
runs, exp_name, colors, line_labels, sw_color, lw_color, net_color, linestyles = (
    load_definitions()
)
# %% calculate means from random samples
mean_ozone = {}
for run in datasets.keys():
    mean_ozone[run] = datasets[run][["o3", "pfull"]].mean(dim="index")

# %%
mean_cf = {}
for run in datasets.keys():
    mean_cf[run] = (
        (datasets[run]["cli"] + datasets[run]["qs"] + datasets[run]["qg"]) > 1e-5
    ).mean(dim="index")

# %% calculate ice weighted ozone
weighted_o3 = {}
for run in datasets.keys():
    cf = (datasets[run]["cli"] + datasets[run]["qs"] + datasets[run]["qg"]) > 1e-5
    weighted_o3[run] = ((cf * datasets[run]["o3"]).sum("height")).mean(dim="index")

cf = (
    datasets["jed0022"]["cli"] + datasets["jed0022"]["qs"] + datasets["jed0022"]["qg"]
) > 1e-5
weighted_control = ((cf * datasets["jed0011"]["o3"]).sum("height")).mean(dim="index")

# %% plot weighted ozone
xvals = {
    "jed0011": 0,
    "jed0022": 2,
    "jed0033": 1,
}
fig, ax = plt.subplots(figsize=(5, 5))
for run in datasets.keys():
    ax.scatter(
        xvals[run],
        weighted_o3[run].values,
        marker="o",
        color='k'
    )
ax.scatter(
    2, weighted_control.values, marker="x", color='k')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(
    [line_labels["jed0011"], line_labels["jed0033"], line_labels["jed0022"]]
)
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylabel("O3 * Cloud Fraction / kg/kg")
fig.savefig(
    "plots/ozone/weighted_ozone_cloud_fraction.png", dpi=300, bbox_inches="tight"
)

# %% plot mean control and 4K in Troposphere
fig, axes = plt.subplots(1, 3, figsize=(10, 6), sharey=True)
#

for run in datasets.keys():
    axes[0].plot(
        mean_ozone[run]["o3"].values.squeeze(),
        mean_ozone[run]["pfull"].values.squeeze() / 100,
        label=run,
        color=colors[run],
    )
    axes[1].plot(
        mean_cf[run].values.squeeze(),
        mean_ozone[run]["pfull"].values.squeeze() / 100,
        label=run,
        color=colors[run],
    )
    axes[2].plot(
        mean_ozone[run]["o3"].values.squeeze()
        * (mean_cf[run].values.squeeze() / mean_cf[run].max().values),
        mean_ozone[run]["pfull"].values.squeeze() / 100,
        label=line_labels[run],
        color=colors[run],
    )

axes[2].plot(
    mean_ozone["jed0011"]["o3"].values.squeeze()
    * (mean_cf["jed0022"].values.squeeze() / mean_cf["jed0022"].max().values),
    mean_ozone["jed0011"]["pfull"].values.squeeze() / 100,
    color='k',
    linestyle="--",
    label="Control O3 * 4K CF",
)

for ax in axes:
    ax.set_ylim(400, 0)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_xlim(4e-8, 1.7e-5)
axes[0].set_xscale("log")
axes[0].set_xlabel("O3 / kg/kg")
axes[1].set_xlabel("Cloud fraction")
axes[2].set_xlabel("O3 * cloud fraction / kg/kg")
axes[0].set_ylabel("Pressure / hPa")
handles, leg_labels = axes[2].get_legend_handles_labels()
fig.legend(
    handles,
    leg_labels,
    bbox_to_anchor=(0.8, -0.01),
    ncol=4,
    frameon=False,
)
fig.tight_layout()
fig.savefig("plots/ozone/profiles_ozone_cf.png", dpi=300, bbox_inches="tight")

# %%
