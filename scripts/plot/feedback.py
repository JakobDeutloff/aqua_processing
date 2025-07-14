# %%
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from src.read_data import read_cloudsat, load_iwp_hists, load_cre, load_random_datasets, load_definitions

# %% load data

datasets = load_random_datasets()
histograms = load_iwp_hists()
cre = load_cre()
runs, exp_name, colors, line_labels, sw_color, lw_color, net_color, linestyles = load_definitions()

# %% plot CRE
fig, ax = plt.subplots(figsize=(7, 4))
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
for run in runs:
    cre[run]['sw'].plot(ax=ax, color=sw_color, label=line_labels[run], linestyle=linestyles[run])
    cre[run]['lw'].plot(ax=ax, color=lw_color, linestyle=linestyles[run])
    cre[run]['net'].plot(ax=ax, color=net_color, linestyle=linestyles[run])

    
labels = ["Net", "SW", "LW", "control", "+2K", "+4K"]
handles = [
    plt.Line2D([0], [0], color=net_color, linestyle="-"),
    plt.Line2D([0], [0], color=sw_color, linestyle="-"),
    plt.Line2D([0], [0], color=lw_color, linestyle="-"),
    plt.Line2D([0], [0], color="grey", linestyle="-."),
    plt.Line2D([0], [0], color="grey", linestyle="--"),
    plt.Line2D([0], [0], color="grey", linestyle="-"),
]

fig.legend(
    handles,
    labels,
    bbox_to_anchor=(0.9, -0.05),
    ncol=6,
    frameon=False,
)

ax.set_xlim([1e-4, 1e1])
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("$I$  / kg m$^{-2}$")
ax.set_ylabel("$C(I)$  / W m$^{-2}$")
ax.set_xscale("log")
fig.savefig(f"plots/feedback/raw/cre.png", dpi=300, bbox_inches="tight")

# %% plot CRE diff
fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)


(cre["jed0033"]["lw"] - cre["jed0011"]["lw"]).plot(
    ax=axes[0], color=lw_color, linestyle=linestyles['jed0033']
)
(cre["jed0022"]["lw"] - cre["jed0011"]["lw"]).plot(
    ax=axes[0], color=lw_color, linestyle=linestyles['jed0022']
)
(cre["jed0033"]["sw"] - cre["jed0011"]["sw"]).plot(
    ax=axes[1], color=sw_color, linestyle=linestyles['jed0033']
)
(cre["jed0022"]["sw"] - cre["jed0011"]["sw"]).plot(
    ax=axes[1], color=sw_color, linestyle=linestyles['jed0022']
)

labels = ["+2 K", "+4 K"]
handles = [
    plt.Line2D([0], [0], color="grey", linestyle="--"),
    plt.Line2D([0], [0], color="grey", linestyle="-"),
]
fig.legend(handles, labels, bbox_to_anchor=[0.65, 0], frameon=True, ncols=2)

for ax in axes:
    ax.set_xlim([1e-4, 10])
    ax.set_ylim([-2, 20])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("$I$  / kg m$^{-2}$")
    ax.set_xscale("log")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)

axes[0].set_ylabel(r"$\Delta C_{\mathrm{lw}}(I)$  / W m$^{-2}$")
axes[1].set_ylabel(r"$\Delta C_{\mathrm{sw}}(I)$  / W m$^{-2}$")
fig.savefig(f"plots/feedback/raw/cre_iwp_diff.png", dpi=300, bbox_inches="tight")

# %% read cloudsat and dardar
cloudsat_raw = read_cloudsat("2008")
dardar_raw = xr.open_dataset("/work/bm1183/m301049/dardar/dardar_iwp_2008.nc")
mask = (dardar_raw['latitude'] > -20) & (dardar_raw['latitude'] < 20) 
dardar_raw = dardar_raw.where(mask)

# %% average over pairs of three entries in cloudsat to get to a resolution of 4.8 km
cloudsat = cloudsat_raw.to_xarray().coarsen({"scnline": 3}, boundary="trim").mean()
dardar = dardar_raw.coarsen({"scanline": 3}, boundary="trim").mean()

# %% calculate iwp hist
iwp_bins = np.logspace(-4, np.log10(40), 51)
histograms["cloudsat"], edges = np.histogram(
    cloudsat["ice_water_path"] / 1e3, bins=iwp_bins, density=False
)
histograms["cloudsat"] = histograms["cloudsat"] / len(cloudsat["ice_water_path"])
histograms["dardar"], _ = np.histogram(
    dardar["iwp"] / 1e3, bins=iwp_bins, density=False
)
histograms["dardar"] = histograms["dardar"] / np.isfinite(dardar['iwp']).sum().values


# %% plot iwp hists
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.stairs(
    histograms["cloudsat"], edges, label="2C-ICE", color="k", linewidth=4, alpha=0.5
)
ax.stairs(
    histograms["dardar"], edges, label="DARDAR", color="brown", linewidth=4, alpha=0.5
)
for run in runs:
    ax.stairs(histograms[run], edges, label=line_labels[run], color=colors[run], linewidth=1.5)

ax.legend()
ax.set_xscale("log")
ax.set_ylim(0, 0.03)
ax.set_ylabel("P($I$)")
ax.set_xlabel("$I$ / kg m$^{-2}$")
ax.set_xlim([1e-4, 40])
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(f"plots/feedback/raw/iwp_hist.png", dpi=300, bbox_inches="tight")

# %% plot diff to control
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
for run in ["jed0022", "jed0033"]:
    diff = histograms[run] - histograms["jed0011"]
    ax.stairs(diff, edges, label=line_labels[run], color=colors[run], linewidth=1.5)

ax.legend()
ax.set_xscale("log")
ax.set_ylabel("$\Delta P(I)$")
ax.set_xlabel("$I$ / kg m$^{-2}$")
ax.set_xlim([1e-4, 40])
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(
    f"plots/feedback/raw/iwp_hist_diff.png", dpi=300, bbox_inches="tight"
)

# %% multiply CRE and iwp hist
cre_folded = {}
const_iwp_folded = {}
const_cre_folded = {}
for run in runs:
    cre_folded[run] = cre[run] * histograms[run]
    const_iwp_folded[run] = cre[run] * histograms["jed0011"]
    const_cre_folded[run] = cre["jed0011"] * histograms[run]

# %% plot folded CRE
fig, ax = plt.subplots(figsize=(7, 4))
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
cre_folded["jed0011"]["net"].plot(ax=ax, color="k", alpha=0.7)
cre_folded["jed0011"]["sw"].plot(ax=ax, color="blue", alpha=0.7)
cre_folded["jed0011"]["lw"].plot(ax=ax, color="red", alpha=0.7)

cre_folded["jed0033"]["net"].plot(ax=ax, color="k", linestyle="-.", alpha=0.7)
cre_folded["jed0033"]["sw"].plot(ax=ax, color="blue", linestyle="-.", alpha=0.7)
cre_folded["jed0033"]["lw"].plot(ax=ax, color="red", linestyle="-.", alpha=0.7)

cre_folded["jed0022"]["net"].plot(ax=ax, color="k", linestyle="--", alpha=0.7)
cre_folded["jed0022"]["sw"].plot(ax=ax, color="blue", linestyle="--", alpha=0.7)
cre_folded["jed0022"]["lw"].plot(ax=ax, color="red", linestyle="--", alpha=0.7)

labels = ["Net", "SW", "LW", "control", "+2K", "+4K"]
handles = [
    plt.Line2D([0], [0], color="k", linestyle="-"),
    plt.Line2D([0], [0], color="blue", linestyle="-"),
    plt.Line2D([0], [0], color="red", linestyle="-"),
    plt.Line2D([0], [0], color="grey", linestyle="-"),
    plt.Line2D([0], [0], color="grey", linestyle="-."),
    plt.Line2D([0], [0], color="grey", linestyle="--"),
]

fig.legend(
    handles,
    labels,
    bbox_to_anchor=(0.9, -0.05),
    ncol=6,
    frameon=False,
)

ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("$I$  / kg m$^{-2}$")
ax.set_ylabel("$C(I) \cdot P(I)$  / W m$^{-2}$")
ax.set_xscale("log")
ax.set_xlim([1e-4, 40])
fig.savefig(f"plots/feedback/raw/cre_iwp_folded.png", dpi=300, bbox_inches="tight")

# %% plot diff of folded CRE
temp_deltas = {"jed0022": 4, "jed0033": 2}
linestyles = {"jed0022": "-", "jed0033": "--"}
fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, sharey=False)
iwp_points = (iwp_bins[1:] + iwp_bins[:-1]) / 2

for run in ["jed0022", "jed0033"]:
    axes[0].plot(
        iwp_points,
        (cre_folded[run]["sw"] - cre_folded["jed0011"]["sw"]) / temp_deltas[run],
        color=sw_color,
        linestyle=linestyles[run],
    )
    axes[0].plot(
        iwp_points,
        (cre_folded[run]["lw"] - cre_folded["jed0011"]["lw"]) / temp_deltas[run],
        color=lw_color,
        linestyle=linestyles[run],
    )
    axes[0].plot(
        iwp_points,
        (cre_folded[run]["net"] - cre_folded["jed0011"]["net"]) / temp_deltas[run],
        color=net_color,
        linestyle=linestyles[run],
    )
    axes[1].plot(
        iwp_points,
        (const_cre_folded[run]["sw"] - const_cre_folded["jed0011"]["sw"])
        / temp_deltas[run],
        color="blue",
        linestyle=linestyles[run],
    )
    axes[1].stairs(
        (const_cre_folded[run]["lw"] - const_cre_folded["jed0011"]["lw"])
        / temp_deltas[run],
        edges,
        color="red",
        linestyle=linestyles[run],
    )
    axes[1].stairs(
        (const_cre_folded[run]["net"] - const_cre_folded["jed0011"]["net"])
        / temp_deltas[run],
        edges,
        color="k",
        linestyle=linestyles[run],
    )
    axes[2].stairs(
        (const_iwp_folded[run]["sw"] - const_iwp_folded["jed0011"]["sw"])
        / temp_deltas[run],
        edges,
        color="blue",
        linestyle=linestyles[run],
    )
    axes[2].stairs(
        (const_iwp_folded[run]["lw"] - const_iwp_folded["jed0011"]["lw"])
        / temp_deltas[run],
        edges,
        color="red",
        linestyle=linestyles[run],
    )
    axes[2].stairs(
        (const_iwp_folded[run]["net"] - const_iwp_folded["jed0011"]["net"])
        / temp_deltas[run],
        edges,
        color="k",
        label=line_labels[run],
        linestyle=linestyles[run],
    )
    axes[2].set_ylabel(r"$F_{\mathrm{CRE}}(I)$ / W m$^{-2}$ K$^{-1}$")
    axes[1].set_ylabel(r"$F_{\mathrm{IWP}}(I)$ / W m$^{-2}$ K$^{-1}$")
    axes[0].set_ylabel(r"$F(I)$ / W m$^{-2}$ K$^{-1}$")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim([1e-4, 40])
    ax.set_xscale("log")
    ax.axhline(0, color="grey", linestyle="-", linewidth=0.8)

axes[-1].set_xlabel("$I$  / kg m$^{-2}$")

labels = ["LW", "SW", "Net", "+2K", "+4K"]
handles = [
    plt.Line2D([0], [0], color="red", linestyle="-"),
    plt.Line2D([0], [0], color="blue", linestyle="-"),
    plt.Line2D([0], [0], color="k", linestyle="-"),
    plt.Line2D([0], [0], color="grey", linestyle="--"),
    plt.Line2D([0], [0], color="grey", linestyle="-"),
]
fig.legend(handles, labels, bbox_to_anchor=[0.8, 0.03], frameon=True, ncols=5)
fig.savefig(
    f"plots/feedback/raw/cre_iwp_folded_diff.png", dpi=300, bbox_inches="tight"
)

# %% calculate integrated CRE and feedback
cre_integrated = {}
cre_const_iwp_integrated = {}
cre_const_cre_integrated = {}
feedback = {}
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

# %% partition IWP feedback into area and opacity feedback 
feedback_area = {}
feedback_opacity = {}
for run in runs[1:]:
    g_cap = ((histograms[run] - histograms['jed0011']).sum() / histograms['jed0011'].sum())
    print(f"g_cap for {run}: {g_cap*100/temp_deltas[run]} %/K")
    g_prime = ((histograms[run] - histograms['jed0011']) / histograms['jed0011']) - g_cap
    feedback_area[run] = cre_integrated['jed0011'] * g_cap / temp_deltas[run]
    feedback_opacity[run] = (g_prime * histograms['jed0011'] * cre['jed0011']).sum() / temp_deltas[run]

# %% plot integrated CRE and feedback
fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True, width_ratios=[1, 1, 1.5])


axes[0].scatter(0, feedback["jed0033"]["lw"].values, color=lw_color, marker="x")
axes[0].scatter(1, feedback["jed0033"]["sw"].values, color=sw_color, marker="x")
axes[0].scatter(2, feedback["jed0033"]["net"].values, color=net_color, marker="x")

axes[0].scatter(
    0, feedback["jed0022"]["lw"].values, color=lw_color, marker="o", facecolors="none"
)
axes[0].scatter(
    1, feedback["jed0022"]["sw"].values, color=sw_color, marker="o", facecolors="none"
)
axes[0].scatter(
    2, feedback["jed0022"]["net"].values, color=net_color, marker="o", facecolors="none"
)
axes[0].set_title("Total")

axes[1].scatter(0, feedback_const_iwp["jed0033"]["lw"].values, color=lw_color, marker="x")
axes[1].scatter(1, feedback_const_iwp["jed0033"]["sw"].values, color=sw_color, marker="x")
axes[1].scatter(2, feedback_const_iwp["jed0033"]["net"].values, color=net_color, marker="x")

axes[1].scatter(
    0,
    feedback_const_iwp["jed0022"]["lw"].values,
    color=lw_color,
    marker="o",
    facecolors="none",
)
axes[1].scatter(
    1,
    feedback_const_iwp["jed0022"]["sw"].values,
    color=sw_color,
    marker="o",
    facecolors="none",
)
axes[1].scatter(
    2,
    feedback_const_iwp["jed0022"]["net"].values,
    color=net_color,
    marker="o",
    facecolors="none",
)
axes[1].set_title("CRE Change")

axes[2].scatter(0, feedback_const_cre["jed0033"]["lw"].values, color=lw_color, marker="x")
axes[2].scatter(1, feedback_const_cre["jed0033"]["sw"].values, color=sw_color, marker="x")
axes[2].scatter(2, feedback_const_cre["jed0033"]["net"].values, color=net_color, marker="x")

axes[2].scatter(
    0,
    feedback_const_cre["jed0022"]["lw"].values,
    color=lw_color,
    marker="o",
    facecolors="none",
)
axes[2].scatter(
    1,
    feedback_const_cre["jed0022"]["sw"].values,
    color=sw_color,
    marker="o",
    facecolors="none",
)
axes[2].scatter(
    2,
    feedback_const_cre["jed0022"]["net"].values,
    color=net_color,
    marker="o",
    facecolors="none",
)
axes[2].scatter(2.5, feedback_area["jed0033"]['net'].values, color="k", marker="x")
axes[2].scatter(
    2.5, feedback_area["jed0022"]['net'].values, color="k", marker="o", facecolors="none"
)
axes[2].scatter(3, feedback_opacity["jed0033"]['net'].values, color="k", marker="x")
axes[2].scatter(
    3, feedback_opacity["jed0022"]['net'].values, color="k", marker="o", facecolors="none"
)
axes[2].set_title("IWP Change")

for ax in axes[:-1]:
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([-0.3, 0, 0.3, 0.6])
    ax.set_xticklabels(["LW", "SW", "Net"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)

axes[2].set_xticks([0, 1, 2, 2.5, 3])
axes[2].set_xticklabels(["LW", "SW", "Net", "Area", "Opacity"])
axes[2].spines[["top", "right"]].set_visible(False)
axes[2].axhline(0, color="grey", linestyle="--", linewidth=0.8)


axes[0].set_ylabel("$F$ / W m$^{-2}$ K$^{-1}$")
axes[1].set_ylabel("$F_{\mathrm{CRE}}$ / W m$^{-2}$ K$^{-1}$")
axes[2].set_ylabel("$F_{\mathrm{IWP}}$ / W m$^{-2}$ K$^{-1}$")

labels = ["+2K", "+4K"]
handles = [
    plt.Line2D([0], [0], color="grey", marker="x", linestyle="none"),
    plt.Line2D([0], [0], color="grey", marker="o", linestyle="none"),
]
fig.legend(handles, labels, bbox_to_anchor=[0.62, 0], frameon=True, ncols=2)
fig.savefig(f"plots/feedback/raw/feedback.png", dpi=300, bbox_inches="tight")


# %%
