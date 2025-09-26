# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from src.read_data import load_random_datasets, load_definitions

# %%
runs, exp_name, colors, line_labels, sw_color, lw_color, net_color, linestyles = (
    load_definitions()
)
iwp_bins = np.logspace(-4, np.log10(40), 51)
iwp_points = (iwp_bins[:-1] + iwp_bins[1:]) / 2
datasets = load_random_datasets()
datasets_raw = {}
for run in runs:
    datasets_raw[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed_64_conn0.nc"
    )
# %% bin lc fraction
lc_binned = {}
lc_binned_raw = {}
for run in runs:
    lc_binned[run] = (
        ((datasets[run]["lwp"] > 1e-4) & (datasets[run]["conn"] == 0))
        .groupby_bins(datasets[run]["iwp"], iwp_bins)
        .mean()
    )
    lc_binned_raw[run] = (
        ((datasets_raw[run]["lwp"] > 1e-4) & (datasets_raw[run]["conn"] == 0))
        .groupby_bins(datasets_raw[run]["iwp"], iwp_bins)
        .mean()
    )

# %% plot lc fraction
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True)
for run in runs:
    axes[0].plot(
        iwp_points,
        lc_binned[run],
        label=line_labels[run],
        color=colors[run],
    )
    axes[1].plot(
        iwp_points,
        lc_binned_raw[run],
        color=colors[run],
    )

for ax in axes:
    ax.set_xlabel("$I$ / $kg m^{-2}$")
    ax.set_xscale("log")
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Liquid Cloud Fraction")
axes[0].set_title("With Tuning Factor")
axes[1].set_title("Without Tuning Factor")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=len(runs),
    bbox_to_anchor=(0.5, -0.15),
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

fig.savefig("plots/publication/lc_frac.pdf", bbox_inches="tight")

# %%
