# %%
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from src.read_data import read_cloudsat

# %% load data
runs = ["jed0011", "jed0022", "jed0033"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
colors = {"jed0011": "k", "jed0022": "red", "jed0033": "orange"}
line_labels = {
    "jed0011": "Control",
    "jed0022": "+4 K",
    "jed0033": "+2 K",
}
datasets = {}
cre_interp_mean = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed.nc"
    )
    cre_interp_mean[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/cre/{run}_cre_raw.nc"
    )

# %% calculate masks
mode = "raw"
mask_type = "raw"
masks_height = {}

if mask_type == "raw":
    for run in runs:
        masks_height[run] = True
elif mask_type == "simple_filter":
    for run in runs:
        masks_height[run] = datasets[run]["hc_top_temperature"] < (273.15 - 35)
elif mask_type == "dist_filter":
    masks_height = {}
    iwp_bins = np.logspace(-4, np.log10(40), 51)
    masks_height["jed0011"] = datasets["jed0011"]["hc_top_temperature"] < datasets[
        "jed0011"
    ]["hc_top_temperature"].where(datasets["jed0011"]["iwp"] > 1e-4).quantile(0.90)
    quantiles = (
        (masks_height["jed0011"] * 1)
        .groupby_bins(datasets["jed0011"]["iwp"], iwp_bins)
        .mean()
    )

    for run in runs[1:]:
        mask = xr.DataArray(
            np.ones_like(datasets[run]["hc_top_temperature"]),
            dims=datasets[run]["hc_top_temperature"].dims,
            coords=datasets[run]["hc_top_temperature"].coords,
        )
        for i in range(len(iwp_bins) - 1):
            mask_ds = (datasets[run]["iwp"] > iwp_bins[i]) & (
                datasets[run]["iwp"] <= iwp_bins[i + 1]
            )
            temp_vals = datasets[run]["hc_top_temperature"].where(mask_ds)
            mask_temp = temp_vals > temp_vals.quantile(quantiles[i])
            # mask n_masked values with the highest temperatures from temp_vals
            mask = xr.where(mask_ds & mask_temp, 0, mask)
        masks_height[run] = mask

# %% plot CRE
fig, ax = plt.subplots(figsize=(7, 4))
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
cre_interp_mean["jed0011"]["net"].plot(ax=ax, color="k", alpha=0.7)
cre_interp_mean["jed0011"]["sw"].plot(ax=ax, color="blue", alpha=0.7)
cre_interp_mean["jed0011"]["lw"].plot(ax=ax, color="red", alpha=0.7)

cre_interp_mean["jed0033"]["net"].plot(ax=ax, color="k", linestyle="-.", alpha=0.7)
cre_interp_mean["jed0033"]["sw"].plot(ax=ax, color="blue", linestyle="-.", alpha=0.7)
cre_interp_mean["jed0033"]["lw"].plot(ax=ax, color="red", linestyle="-.", alpha=0.7)

cre_interp_mean["jed0022"]["net"].plot(ax=ax, color="k", linestyle="--", alpha=0.7)
cre_interp_mean["jed0022"]["sw"].plot(ax=ax, color="blue", linestyle="--", alpha=0.7)
cre_interp_mean["jed0022"]["lw"].plot(ax=ax, color="red", linestyle="--", alpha=0.7)

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

ax.set_xlim([1e-4, 1e1])
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("$I$  / kg m$^{-2}$")
ax.set_ylabel("$C(I)$  / W m$^{-2}$")
ax.set_xscale("log")
fig.savefig(f"plots/feedback/{mode}/cre_iwp.png", dpi=300, bbox_inches="tight")

# %% plot CRE diff
fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)


(cre_interp_mean["jed0033"]["lw"] - cre_interp_mean["jed0011"]["lw"]).plot(
    ax=axes[0], color="orange", alpha=0.7, linestyle="-"
)
(cre_interp_mean["jed0022"]["lw"] - cre_interp_mean["jed0011"]["lw"]).plot(
    ax=axes[0], color="r", alpha=0.7, linestyle="-"
)
(cre_interp_mean["jed0033"]["sw"] - cre_interp_mean["jed0011"]["sw"]).plot(
    ax=axes[1], color="orange", alpha=0.7, linestyle="-"
)
(cre_interp_mean["jed0022"]["sw"] - cre_interp_mean["jed0011"]["sw"]).plot(
    ax=axes[1], color="r", alpha=0.7, linestyle="-"
)

labels = ["+2 K", "+4 K"]
handles = [
    plt.Line2D([0], [0], color="r", linestyle="-"),
    plt.Line2D([0], [0], color="orange", linestyle="-"),
]
fig.legend(handles, labels, bbox_to_anchor=[0.65, 0], frameon=True, ncols=2)

for ax in axes:
    ax.set_xlim([1e-4, 10])
    ax.set_ylim([-2, 20])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("$I$  / kg m$^{-2}$")
    ax.set_xscale("log")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)

axes[0].set_ylabel(r"$\Delta C_{\mathrm{lw}}(I)$  / W m$^{-2}$", color='red')
axes[1].set_ylabel(r"$\Delta C_{\mathrm{sw}}(I)$  / W m$^{-2}$", color='blue')
fig.savefig(f"plots/feedback/{mode}/cre_iwp_diff.png", dpi=300, bbox_inches="tight")

# %% read cloudsat
cloudsat_raw = read_cloudsat("2009")

# %% average over pairs of three entries in cloudsat to get to a resolution of 4.8 km
cloudsat = cloudsat_raw.to_xarray().coarsen({"scnline": 3}, boundary="trim").mean()

# %% calculate iwp hist
iwp_bins = np.logspace(-4, np.log10(40), 51)
histograms = {}
for run in runs:
    iwp = datasets[run]["iwp"].where(masks_height[run])
    histograms[run], edges = np.histogram(iwp, bins=iwp_bins, density=False)
    histograms[run] = histograms[run] / len(iwp)

histograms["cloudsat"], _ = np.histogram(
    cloudsat["ice_water_path"] / 1e3, bins=iwp_bins, density=False
)
histograms["cloudsat"] = histograms["cloudsat"] / len(cloudsat["ice_water_path"])

# %% plot iwp hists
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.stairs(
    histograms["cloudsat"], edges, label="2C-ICE", color="k", linewidth=4, alpha=0.5
)
for run in ['jed0011']:
    ax.stairs(histograms[run], edges, label=line_labels[run], color=colors[run])

ax.legend()
ax.set_xscale("log")
ax.set_ylim(0, 0.02)
ax.set_ylabel("P($I$)")
ax.set_xlabel("$I$ / kg m$^{-2}$")
ax.set_xlim([1e-4, 40])
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(f"plots/feedback/{mode}/iwp_hist_cont.png", dpi=300, bbox_inches="tight")

# %% plot diff to control
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
for run in ["jed0022", "jed0033"]:
    diff = histograms[run] - histograms["jed0011"]
    ax.stairs(diff, edges, label=line_labels[run], color=colors[run])

ax.legend()
ax.set_xscale("log")
ax.set_ylabel("$\Delta P(I)$")
ax.set_xlabel("$I$ / kg m$^{-2}$")
ax.set_xlim([1e-4, 40])
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(
    f"plots/feedback/{mode}/iwp_hist_diff.png", dpi=300, bbox_inches="tight"
)
# %% multiply CRE and iwp hist
cre_folded = {}
const_iwp_folded = {}
const_cre_folded = {}
for run in runs:
    cre_folded[run] = cre_interp_mean[run] * histograms[run]
    const_iwp_folded[run] = cre_interp_mean[run] * histograms["jed0011"]
    const_cre_folded[run] = cre_interp_mean["jed0011"] * histograms[run]

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
fig.savefig(f"plots/feedback/{mode}/cre_iwp_folded.png", dpi=300, bbox_inches="tight")

# %% plot diff of folded CRE
temp_deltas = {"jed0022": 4, "jed0033": 2}
linestyles = {"jed0022": "-", "jed0033": "--"}
fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, sharey=False)

for run in ["jed0022", "jed0033"]:
    axes[0].stairs(
        (cre_folded[run]["sw"] - cre_folded["jed0011"]["sw"]) / temp_deltas[run],
        edges,
        color="blue",
        linestyle=linestyles[run],
    )
    axes[0].stairs(
        (cre_folded[run]["lw"] - cre_folded["jed0011"]["lw"]) / temp_deltas[run],
        edges,
        color="red",
        linestyle=linestyles[run],
    )
    axes[0].stairs(
        (cre_folded[run]["net"] - cre_folded["jed0011"]["net"]) / temp_deltas[run],
        edges,
        color="k",
        linestyle=linestyles[run],
    )
    axes[1].stairs(
        (const_cre_folded[run]["sw"] - const_cre_folded["jed0011"]["sw"])
        / temp_deltas[run],
        edges,
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
        label=exp_name[run],
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
    f"plots/feedback/{mode}/cre_iwp_folded_diff.png", dpi=300, bbox_inches="tight"
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

# %% plot integrated CRE and feedback
fig, axes = plt.subplots(1, 3, figsize=(9, 4), sharex=True, sharey=True)


axes[0].scatter(0, feedback["jed0033"]["lw"].values, color="red", marker="x")
axes[0].scatter(1, feedback["jed0033"]["sw"].values, color="blue", marker="x")
axes[0].scatter(2, feedback["jed0033"]["net"].values, color="k", marker="x")

axes[0].scatter(
    0, feedback["jed0022"]["lw"].values, color="red", marker="o", facecolors="none"
)
axes[0].scatter(
    1, feedback["jed0022"]["sw"].values, color="blue", marker="o", facecolors="none"
)
axes[0].scatter(
    2, feedback["jed0022"]["net"].values, color="k", marker="o", facecolors="none"
)
axes[0].set_title("Total")

axes[1].scatter(0, feedback_const_iwp["jed0033"]["lw"].values, color="red", marker="x")
axes[1].scatter(1, feedback_const_iwp["jed0033"]["sw"].values, color="blue", marker="x")
axes[1].scatter(2, feedback_const_iwp["jed0033"]["net"].values, color="k", marker="x")

axes[1].scatter(
    0,
    feedback_const_iwp["jed0022"]["lw"].values,
    color="red",
    marker="o",
    facecolors="none",
)
axes[1].scatter(
    1,
    feedback_const_iwp["jed0022"]["sw"].values,
    color="blue",
    marker="o",
    facecolors="none",
)
axes[1].scatter(
    2,
    feedback_const_iwp["jed0022"]["net"].values,
    color="k",
    marker="o",
    facecolors="none",
)
axes[1].set_title("CRE Change")

axes[2].scatter(0, feedback_const_cre["jed0033"]["lw"].values, color="red", marker="x")
axes[2].scatter(1, feedback_const_cre["jed0033"]["sw"].values, color="blue", marker="x")
axes[2].scatter(2, feedback_const_cre["jed0033"]["net"].values, color="k", marker="x")

axes[2].scatter(
    0,
    feedback_const_cre["jed0022"]["lw"].values,
    color="red",
    marker="o",
    facecolors="none",
)
axes[2].scatter(
    1,
    feedback_const_cre["jed0022"]["sw"].values,
    color="blue",
    marker="o",
    facecolors="none",
)
axes[2].scatter(
    2,
    feedback_const_cre["jed0022"]["net"].values,
    color="k",
    marker="o",
    facecolors="none",
)
axes[2].set_title("IWP Change")

for ax in axes:
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([-0.3, 0, 0.3, 0.6])
    ax.set_xticklabels(["LW", "SW", "Net"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)


axes[0].set_ylabel("$F$ / W m$^{-2}$ K$^{-1}$")

labels = ["+2K", "+4K"]
handles = [
    plt.Line2D([0], [0], color="grey", marker="x", linestyle="none"),
    plt.Line2D([0], [0], color="grey", marker="o", linestyle="none"),
]
fig.legend(handles, labels, bbox_to_anchor=[0.62, 0], frameon=True, ncols=2)
fig.savefig(f"plots/feedback/{mode}/feedback.png", dpi=300, bbox_inches="tight")


# %%
