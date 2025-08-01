# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from src.read_data import load_random_datasets, load_vgrid, load_definitions

# %% load CRE data
runs, exp_name, colors, line_labels, sw_color, lw_color, net_color, linestyles = (
    load_definitions()
)
iwp_bins = np.logspace(-4, np.log10(40), 51)
datasets = load_random_datasets()
vgrid = load_vgrid()
T_delta = {
    "jed0011": 0,
    "jed0022": 4,
    "jed0033": 2,
}

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
        density=False,
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
        density=False,
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

# %% get normalised incoming SW for every bin 
SW_in = datasets['jed0011']['rsdt'].groupby_bins(
    datasets['jed0011']['time_local'], bins=bins).mean()
SW_in = SW_in / SW_in.max()

# %% plot
fig, axes = plt.subplots(2, 2, figsize=(12, 6), width_ratios=[1, 0.5])


for run in runs:
    axes[0, 0].stairs(
        histograms_iwp[run],
        edges,
        label=exp_name[run],
        color=colors[run],
        alpha=0.5
    )
    axes[0, 0].plot(
        x, sin_func(x, *parameters_iwp[run]), color=colors[run], linestyle="--"
    )

    axes[1, 0].stairs(
        histograms_wa[run],
        edges,
        label=exp_name[run],
        color=colors[run],
        alpha=0.5
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
    plt.Line2D([0], [0], color=colors["jed0011"], linestyle="-"),
    plt.Line2D([0], [0], color=colors['jed0033'], linestyle="-"),
    plt.Line2D([0], [0], color=colors['jed0022'], linestyle="-"),
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

# %% try alternative plot 
fig, axes = plt.subplots(2, 2, figsize=(12, 6), width_ratios=[1, 0.5])


for run in runs:
    axes[0, 0].plot(
        x,
        histograms_iwp[run]*SW_in,
        label=exp_name[run],
        color=colors[run],
        linewidth=2,
    )

    axes[1, 0].plot(
        x, 
        histograms_wa[run]*SW_in,
        label=exp_name[run],
        color=colors[run],
        linewidth=2,
    )

for ax in axes[:, 0]:
    ax.set_xlim([5.9, 18.1])
    ax.set_ylim([0, 0.05])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([6, 12, 18])
    ax.set_yticks([0, 0.03, 0.05])
axes[1, 0].set_xlabel("Local Time / h")
axes[0, 0].set_ylabel("P($I$ > 1 kg m$^{-2}$) $\cdot ~ S$")
axes[1, 0].set_ylabel(r"P($\omega$ > 1 m s$^{-1}$) $\cdot ~ S$")

for run in runs:
    axes[0, 1].scatter(
        T_delta[run],
        (histograms_iwp[run] * SW_in).sum(),
        color=colors[run],
        marker='o',
    )
    axes[0, 1].scatter(
        T_delta[run],
        (histograms_wa[run] * SW_in).sum(),
        color=colors[run],
        marker='x',
    )
axes[0, 1].set_xticks([0, 2, 4])
axes[0, 1].set_xticklabels(["Control", "+2 K", "+4 K"])
axes[0, 1].set_ylabel("Daily Maximum / h")
axes[0, 1].spines[["top", "right"]].set_visible(False)

axes[1, 1].remove()

# add legend 
handles = [
    plt.Line2D([0], [0], color=colors["jed0011"], linestyle="-"),
    plt.Line2D([0], [0], color=colors['jed0033'], linestyle="-"),
    plt.Line2D([0], [0], color=colors['jed0022'], linestyle="-"),
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

fig.savefig("plots/publication/figure3_alt.pdf", bbox_inches="tight")
# %% calculate day night occurence 
day_occurrence_iwp = {}
day_occurence_vel = {}
for run in runs:
    day_occurrence_iwp[run] = datasets[run]["time_local"].where(
        (datasets[run]["time_local"] >= 6) & (datasets[run]["time_local"] < 18) & (datasets[run]["iwp"] > 1e0)
    ).count() / datasets[run]["time_local"].where(datasets[run]['iwp'] > 1e0).count()

    vels = (
        datasets[run]["wa"].isel(height_2=slice(height_2, height_1)).max(dim="height_2")
    )
    mask = vels > 1
    day_occurence_vel[run] = datasets[run]["time_local"].where(
        (datasets[run]["time_local"] >= 6) & (datasets[run]["time_local"] < 18) & (mask)
    ).count() / datasets[run]["time_local"].where(mask).count()



# %% plot day night occurence
fig, ax = plt.subplots(figsize=(4, 4))
for run in runs:
    ax.scatter(
        T_delta[run],
        day_occurrence_iwp[run]*100,
        color=colors[run],
        marker='o',
        label=f"{exp_name[run]} IWP"
    )
    ax.scatter(
        T_delta[run],
        day_occurence_vel[run]*100,
        color=colors[run],
        marker='x',
        label=f"{exp_name[run]} WA"
    )
ax.set_xticks([0, 2, 4])
ax.set_xticklabels(["Control", "+2 K", "+4 K"])
ax.set_ylabel("P(Occurrence during Day) / %")
handles = [
    plt.Line2D([0], [0], color='grey', marker='o', linestyle=''),
    plt.Line2D([0], [0], color='grey', marker='x', linestyle=''),
]
labels = ["$I > 1$ kg m$^{-2}$", r"$\omega > 1$ m s$^{-1}$"]
fig.legend(
    handles,
    labels,
    bbox_to_anchor=(0.88, -0.05),
    ncols=2,
    frameon=False,
)
ax.spines[["top", "right"]].set_visible(False)
# %%
