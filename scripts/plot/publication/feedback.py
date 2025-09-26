# %%
import matplotlib.pyplot as plt
import numpy as np
from src.read_data import (
    load_iwp_hists,
    load_cre,
    load_random_datasets,
    load_definitions,
)

# mpl.use("Webagg")
# %% load data
temp_deltas = {"jed0022": 4, "jed0033": 2}
iwp_bins = np.logspace(-4, np.log10(40), 51)
iwp_points = (iwp_bins[:-1] + iwp_bins[1:]) / 2
runs, exp_name, colors, line_labels, sw_color, lw_color, net_color, linestyles = (
    load_definitions()
)
xpos = {
    "lw": 0,
    "sw": 1,
    "net": 2,
}
facecolors = {
    "jed0022": "none",
    "jed0033": None,
}
markers = {
    "jed0022": "o",
    "jed0033": "x",
}
colors_fluxes = {
    'lw': lw_color,
    'sw': sw_color,
    'net': net_color,
}
datasets = load_random_datasets()
histograms = load_iwp_hists()
cre = load_cre()
# %% multiply CRE and iwp hist
cre_folded = {}
const_iwp_folded = {}
const_cre_folded = {}
for run in runs:
    cre_folded[run] = cre[run] * histograms[run]
    const_iwp_folded[run] = cre[run] * histograms["jed0011"]
    const_cre_folded[run] = cre["jed0011"] * histograms[run]
# %% calculate integrated CRE and feedback
cre_integrated = {}
cre_const_iwp_integrated = {}
cre_const_cre_integrated = {}
feedback = {}
feedback_linear = {}
feedback_const_iwp = {}
feedback_const_cre = {}

for run in runs:
    cre_integrated[run] = cre_folded[run].sum(dim="iwp")
    cre_const_iwp_integrated[run] = const_iwp_folded[run].sum(dim="iwp")
    cre_const_cre_integrated[run] = const_cre_folded[run].sum(dim="iwp")


feedback["jed0033"] = (cre_integrated["jed0033"] - cre_integrated["jed0011"]) / 2
feedback["jed0022"] = (cre_integrated["jed0022"] - cre_integrated["jed0011"]) / 4
feedback_const_iwp["jed0033"] = (
    cre_const_iwp_integrated["jed0033"] - cre_const_iwp_integrated["jed0011"]
) / 2
feedback_const_iwp["jed0022"] = (
    cre_const_iwp_integrated["jed0022"] - cre_const_iwp_integrated["jed0011"]
) / 4
feedback_const_cre["jed0033"] = (
    cre_const_cre_integrated["jed0033"] - cre_const_cre_integrated["jed0011"]
) / 2
feedback_const_cre["jed0022"] = (
    cre_const_cre_integrated["jed0022"] - cre_const_cre_integrated["jed0011"]
) / 4
feedback_linear["jed0033"] = (
    feedback_const_cre["jed0033"] + feedback_const_iwp["jed0033"]
)
feedback_linear["jed0022"] = (
    feedback_const_cre["jed0022"] + feedback_const_iwp["jed0022"]
)
# %% partition IWP feedback into area and opacity feedback
feedback_area = {}
feedback_opacity = {}
for run in runs[1:]:
    g_cap = (histograms[run] - histograms["jed0011"]).sum() / histograms[
        "jed0011"
    ].sum()
    print(f"g_cap for {run}: {g_cap*100/temp_deltas[run]} %/K")
    g_prime = (
        (histograms[run] - histograms["jed0011"]) / histograms["jed0011"]
    ) - g_cap
    feedback_area[run] = cre_integrated["jed0011"] * g_cap / temp_deltas[run]
    feedback_opacity[run] = (
        g_prime * histograms["jed0011"] * cre["jed0011"]
    ).sum() / temp_deltas[run]

# %% plot
fig, axes = plt.subplots(
    3, 3, figsize=(10, 7), width_ratios=[3, 1, 0.3], constrained_layout=True
)


for run in runs[1:]:

    for flux in ["lw", "sw", "net"]:
        # IWP resolved Feedback
        axes[0, 0].plot(
            iwp_points,
            (
                const_iwp_folded[run][flux].values
                - const_iwp_folded["jed0011"][flux].values
            )
            / temp_deltas[run],
            color=colors_fluxes[flux],
            linestyle=linestyles[run],
        )
        axes[1, 0].plot(
            iwp_points,
            (
                const_cre_folded[run][flux].values
                - const_cre_folded["jed0011"][flux].values
            )
            / temp_deltas[run],
            color=colors_fluxes[flux],
            linestyle=linestyles[run],
        )
        axes[2, 0].plot(
            iwp_points,
            (cre_folded[run][flux].values - cre_folded["jed0011"][flux].values)
            / temp_deltas[run],
            color=colors_fluxes[flux],
            linestyle=linestyles[run],
        )
        # Integrated Feedback
        axes[0, 1].scatter(
            xpos[flux],
            feedback_const_iwp[run][flux].values,
            color=colors_fluxes[flux],
            marker=markers[run],
            facecolor=facecolors[run],
        )
        axes[1, 1].scatter(
            xpos[flux],
            feedback_const_cre[run][flux].values,
            color=colors_fluxes[flux],
            marker=markers[run],
            facecolor=facecolors[run],
        )
        axes[2, 1].scatter(
            xpos[flux],
            feedback_linear[run][flux].values,
            color=colors_fluxes[flux],
            marker=markers[run],
            facecolor=facecolors[run],
        )

    # plot area and opacity feedback
    axes[1, 2].scatter(
        0,
        feedback_area[run]["net"].values,
        color=colors_fluxes["net"],
        marker=markers[run],
        facecolor=facecolors[run],
    )
    axes[1, 2].scatter(
        1,
        feedback_opacity[run]["net"].values,
        color=colors_fluxes["net"],
        marker=markers[run],
        facecolor=facecolors[run],
    )


axes[0, 0].set_ylabel(r"$F_{\mathrm{CRE}}(I)$ / W m$^{-2}$ K$^{-1}$")
axes[1, 0].set_ylabel(r"$F_{\mathrm{IWP}}(I)$ / W m$^{-2}$ K$^{-1}$")
axes[2, 0].set_ylabel(r"$F(I)$ / W m$^{-2}$ K$^{-1}$")
axes[2, 0].set_xlabel(r"$I$ / kg m$^{-2}$")
axes[0, 1].set_ylabel(r"$F_{\mathrm{CRE}}$ / W m$^{-2}$ K$^{-1}$")
axes[1, 1].set_ylabel(r"$F_{\mathrm{IWP}}$ / W m$^{-2}$ K$^{-1}$")
axes[2, 1].set_ylabel(r"$F$ / W m$^{-2}$ K$^{-1}$")

# make legends
labels = [
    'LW', 'SW', 'Net', "+2 K", "+4 K"
]
handles = [
    plt.Line2D([0], [0], color=lw_color),
    plt.Line2D([0], [0], color=sw_color),
    plt.Line2D([0], [0], color=net_color),
    plt.Line2D([0], [0], color="grey", linestyle=linestyles["jed0033"]),
    plt.Line2D([0], [0], color="grey", linestyle=linestyles["jed0022"]),
]

fig.legend(
    handles,
    labels,
    ncol=5,
    bbox_to_anchor=(0.6, 0),
    frameon=False,
)

handles = [
    axes[2, 2].scatter([0], [0], color="grey", marker="x", linestyle=""),
    axes[2, 2].scatter(
        [0], [0], color="grey", marker="o", linestyle="", facecolors="none"
    ),
]
labels = [
    "+2 K",
    "+4 K",
]
fig.legend(
    handles,
    labels,
    ncol=2,
    bbox_to_anchor=(0.73, 0),
    loc="upper left",
    frameon=False,
)

# configure axes
axes[0, 2].remove()
axes[2, 2].remove()

for ax in axes[:, 0]:
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 20)
    ax.set_ylim([-0.025, 0.045])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_yticks([-0.02, 0, 0.02, 0.04])

for ax in axes[:, 1]:
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_ylim([-0.5, 0.7])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_yticks([-0.3, 0, 0.3, 0.6])
    ax.set_xticks([0, 1, 2])
    ax.set_xlim([-0.3, 2.3])
    ax.set_xticklabels(["LW", "SW", "NET"])

axes[1, 2].set_ylim([-0.5, 0.7])
axes[1, 2].set_yticks([-0.3, 0, 0.3, 0.6])
axes[1, 2].set_yticklabels([])
axes[1, 2].set_xticks([0, 1])
axes[1, 2].set_xlim([-0.2, 1.2])
axes[1, 2].spines[["top", "right"]].set_visible(False)
axes[1, 2].axhline(0, color="k", linewidth=0.5)
axes[1, 2].set_xticklabels(["Area", "Opacity"], rotation=-45)

# add letters
a = list(axes[:, 0:2].flatten())
a.insert(4, axes[1, 2])
for i, ax in enumerate(a):
    ax.text(
        0.03,
        1,
        chr(97 + i),
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
    )


fig.savefig("plots/publication/feedback.pdf", bbox_inches="tight")
plt.show()
# %% non -linearity of feedback
feedback_nonlin = {}
for run in runs[1:]:
    flux_feedback = {}
    for flux in ["lw", "sw", "net"]:
        flux_feedback[flux] = (
            feedback_const_cre[run][flux] + feedback_const_iwp[run][flux]
        ) - feedback[run][flux]
    feedback_nonlin[run] = flux_feedback


# %%
