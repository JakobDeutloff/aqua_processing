# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from src.read_data import load_random_datasets, load_vgrid

# %% load CRE data
runs = ["jed0011", "jed0033", "jed0022"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
colors = colors = {"jed0011": "#3e0237", "jed0022": "#f707da", "jed0033": "#9a0488"}
T_delta = {
    "jed0011": 0,
    "jed0022": 4,
    "jed0033": 2,
}
line_labels = {
    "jed0011": "Control",
    "jed0022": "+4 K",
    "jed0033": "+2 K",
}
iwp_bins = np.logspace(-4, np.log10(40), 51)
datasets = load_random_datasets()
vgrid = load_vgrid()


# %% fit sin to daily cycle
def sin_func(x, a, b, c):
    return a * np.sin((x * 2 * np.pi / 24) + (b * 2 * np.pi / 24)) + c


parameters_iwp = {}
histograms_iwp = {}
parameters_wa = {}
histograms_wa = {}
quantiles = {}
bins = np.arange(0, 25, 1)
height_1 = np.abs((vgrid["zghalf"] - 2e3)).argmin(dim="height_2").values
height_2 = np.abs((vgrid["zghalf"] - 15e3)).argmin(dim="height_2").values
x = (bins[:-1] + bins[1:]) / 2
for run in runs:
    histograms_iwp[run], edges = np.histogram(
        datasets[run]["time_local"].where(datasets[run]["iwp"] > 1e0),
        bins,
        density=True,
    )
    options = {"ftol": 1e-15}
    popt, pcov = curve_fit(
        sin_func,
        x,
        histograms_iwp[run],
        p0=[0.00517711, 1, 0.30754356],
        method="trf",
        **options
    )
    parameters_iwp[run] = popt

    vels = (
        datasets[run]["wa"].isel(height_2=slice(height_2, height_1)).max(dim="height_2")
    )
    mask = vels > 1

    histograms_wa[run], edges = np.histogram(
        datasets[run]["time_local"].where(mask),
        bins,
        density=True,
    )
    popt, pcov = curve_fit(
        sin_func,
        x,
        histograms_wa[run],
        p0=[0.00517711, 1, 0.30754356],
        method="trf",
        **options
    )
    parameters_wa[run] = popt

# %% plot
fig, axes = plt.subplots(2, 2, figsize=(12, 6), width_ratios=[1, 0.5])


for run in runs:
    axes[0, 0].stairs(
        histograms_iwp[run],
        edges,
        label=exp_name[run],
        color=colors[run],
    )
    axes[0, 0].plot(
        x, sin_func(x, *parameters_iwp[run]), color=colors[run], linestyle="--"
    )

    axes[1, 0].stairs(
        histograms_wa[run],
        edges,
        label=exp_name[run],
        color=colors[run],
    )
    axes[1, 0].plot(
        x, sin_func(x, *parameters_wa[run]), color=colors[run], linestyle="--"
    )

for ax in axes[:, 0]:
    ax.set_xlim([0.1, 23.9])
    ax.set_ylim([0.03, 0.06])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([6, 12, 18])
    ax.set_yticks([0.03, 0.04, 0.05, 0.06])
axes[1, 0].set_xlabel("Local Time / h")
axes[0, 0].set_ylabel("P($I$ > 1 kg m$^{-2}$)")
axes[1, 0].set_ylabel(r"P($\omega$ > 1 m s$^{-1}$)")

for run in runs:
    axes[0, 1].scatter(
        T_delta[run],
        6 - parameters_iwp[run][1],
        color=colors[run],
        marker='o',
    )
    axes[0, 1].scatter(
        T_delta[run],
        6 - parameters_wa[run][1],
        color=colors[run],
        marker='x',
    )
axes[0, 1].set_xticks([0, 2, 4])
axes[0, 1].set_xticklabels(["Control", "+2 K", "+4 K"])
axes[0, 1].set_ylabel("Daily Maximum / h")
axes[0, 1].spines[["top", "right"]].set_visible(False)
axes[0, 1].set_yticks([4, 6])

axes[1, 1].remove()

# add legend 
handles = [
    plt.Line2D([0], [0], color="midnightblue", linestyle="-"),
    plt.Line2D([0], [0], color="darkviolet", linestyle="-"),
    plt.Line2D([0], [0], color="deeppink", linestyle="-"),
    plt.Line2D([0], [0], color="grey", marker="o", linestyle=""),
    plt.Line2D([0], [0], color="grey", marker="x", linestyle=""),
]
labels = ["Control", "+2 K", "+4 K", "$I > 1$ kg m$^{-2}$", r"$\omega > 1$ m s$^{-1}$"]
fig.legend(
    handles,
    labels,
    bbox_to_anchor=(0.88, 0.4),
    ncols=2,
    frameon=False,
)

# add letters 
for ax, letter in zip([axes[0, 0], axes[1, 0], axes[0, 1]], ["a", "b", "c"]):
    ax.text(
        0.03,
        1,
        letter,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
    )

fig.savefig("plots/publication/figure3.pdf", bbox_inches="tight")

# %%
