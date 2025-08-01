# %%
import matplotlib.pyplot as plt
from src.read_data import load_random_datasets, load_vgrid, load_definitions
from src.calc_variables import calculate_hc_temperature_bright
import numpy as np

# %% load data
runs, exp_name, colors, line_labels, sw_color, lw_color, net_color, linestyles = (
    load_definitions()
)
datasets = load_random_datasets()
vgrid = load_vgrid()

# %% calculate hc temp with brightness temperature
hc_temps = {}
for run in runs:
    print(run)
    temp, p = calculate_hc_temperature_bright(
        datasets[run],
        vgrid["zg"],
    )
    datasets[run] = datasets[run].assign({"hc_top_temp_bright": temp, "hc_top_pressure_bright": p})


# %% bin temperature
iwp_bins = np.logspace(-4, np.log10(40), 51)
binned_temp = {}
for run in runs:
    binned_temp[run] = (
        datasets[run]["hc_top_temp_bright"]
        .groupby_bins(datasets[run]["iwp"], iwp_bins)
        .mean()
    )

# %% plot hc_temp binned by IWP
fig, axes = plt.subplots(2, 1, figsize=(6, 4), sharex="col", height_ratios=[3, 1.5])

for run in runs:
    binned_temp[run].plot(
        ax=axes[0],
        color=colors[run],
        label=line_labels[run],
    )

axes[1].axhline(0, color="k", lw=0.5)
for run in runs[1:]:
    (binned_temp[run] - binned_temp[runs[0]]).plot(
        ax=axes[1],
        color=colors[run],
        label=line_labels[run],
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
    ncols=3,
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
fig.savefig(
    'plots/publication/figure2.pdf', bbox_inches="tight"
)

# %% calculate mean hc temperatures
means = {}
for run in runs:
    means[run] = datasets[run]["hc_top_temp_bright"].where(datasets[run]["iwp"] > 1e-4).mean().values
    print(f"mean hc temp {line_labels[run]} {means[run]:.2f} K")
# %%
