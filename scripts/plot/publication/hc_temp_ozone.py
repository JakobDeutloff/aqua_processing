# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from src.read_data import load_random_datasets, load_definitions, load_vgrid, load_cre, load_iwp_hists

# %% load CRE data
runs, exp_name, colors, line_labels, sw_color, lw_color, net_color, linestyles = (
    load_definitions()
)
iwp_bins = np.logspace(-4, np.log10(40), 51)
iwp_points = (iwp_bins[:-1] + iwp_bins[1:]) / 2
datasets = load_random_datasets()
cre = load_cre()
histograms = load_iwp_hists()
vgrid = load_vgrid()
datasets['jed2224'] = xr.open_dataset(
    "/work/bm1183/m301049/icon_hcap_data/const_o3/production/random_sample/jed2224_randsample_processed_64.nc"
)
cre['jed2224'] = xr.open_dataset(
    "/work/bm1183/m301049/icon_hcap_data/const_o3/production/cre/jed2224_cre_raw.nc"
)
colors["jed2224"] = 'k'
line_labels["jed2224"] = "+4 K const. O$_3$"
linestyles = {
    "jed0011": "-",
    "jed0022": "-",
    "jed0033": "-",
    "jed2224": "--",
}
runs = runs + ["jed2224"]

# %% calculate binned T_hc
temp_binned = {}
for run in runs:
    temp_binned[run] = (
        datasets[run]["hc_top_temperature"]
        .groupby_bins(datasets[run]["iwp"], iwp_bins)
        .mean()
    )

# %% plot hc_temp binned by IWP

fig, axes = plt.subplots(2, 1, figsize=(5, 3.5), sharex="col", height_ratios=[3, 1.5])

for run in runs:
    temp_binned[run].plot(
        ax=axes[0],
        color=colors[run],
        label=line_labels[run],
        linestyle=linestyles[run],
    )


axes[1].axhline(0, color="k", lw=0.5)
for run in runs[1:]:
    (temp_binned[run] - temp_binned[runs[0]]).plot(
        ax=axes[1],
        color=colors[run],
        label=line_labels[run],
        linestyle=linestyles[run],
    )

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("")
    ax.set_xscale("log")
    ax.set_xlim([1e-4, 20])

axes[0].set_ylabel(r"T$_{\mathrm{hc}}(I)$ / K")
axes[0].set_yticks([200, 215, 230])
axes[0].set_ylim([195, 235])
axes[1].set_ylabel(r"$\Delta$ T$_{\mathrm{hc}}(I)$ / K")
axes[1].set_yticks([-1, 0, 1])
axes[1].set_ylim([-1, 1])
axes[1].set_xlabel(r"$I$ / kg m$^{-2}$")

handles, names = axes[0].get_legend_handles_labels()

fig.legend(
    handles=handles,
    labels=names,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.15),
    ncols=4,
    frameon=False,
)

# add letters 
for ax, letter in zip(axes, ["a", "b"]):
    ax.text(
        0.03,
        1,
        letter,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
    )

fig.savefig("plots/publication/hc_temp_ozone.pdf", bbox_inches="tight")
# %% calculate lw feedback 
lw_feedback = {}
temp_deltas = {
    "jed0011": 0,
    "jed0022": 4,
    "jed0033": 2,
    "jed2224": 4,
}
for run in runs[1:]:
    lw_feedback[run] = (
        (cre[run]["lw"] - cre[runs[0]]["lw"]) * histograms[runs[0]]
    ).sum().values / temp_deltas[run]

print(f"LW feedback reduced by {(lw_feedback['jed2224'] - lw_feedback['jed0022']):.2f} W m$^{-2}$ K$^{-1}$ by interactive ozone from {lw_feedback['jed0022']:.2f} to {lw_feedback['jed2224']:.2f} W m$^{-2}$ K$^{-1}$")

# %% calculate temperature increase in k/k 
means = {}
temp_deltas = {'jed0022': 4, 'jed0033': 2, 'jed2224': 4}
for run in runs:
    means[run] = datasets[run]["hc_top_temperature"].where(datasets[run]["iwp"] > 1e-4).mean().values

for run in runs[1:]:
    print(
        f"{run}: {((means[run] - means[runs[0]]) / temp_deltas[run]):.2f}  K/K"
    )
# %%
