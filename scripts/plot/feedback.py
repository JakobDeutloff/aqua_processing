# %%
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from src.read_data import read_cloudsat

# %% load data
runs = ["jed0011", "jed0022", "jed0033"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
datasets = {}
cre_interp_mean = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed.nc"
    )
    cre_interp_mean[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/cre/{run}_cre_interp_mean_rand.nc"
    )


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

ax.set_xlim([1e-4, 2e1])
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("$I$  / kg m$^{-2}$")
ax.set_ylabel("$C(I)$  / W m$^{-2}$")
ax.set_xscale("log")
fig.savefig("plots/cre_iwp_4K.png", dpi=300, bbox_inches="tight")

# %% plot CRE diff
fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)


(cre_interp_mean["jed0033"]["lw"] - cre_interp_mean["jed0011"]["lw"]).plot(
    ax=axes[0], color="r", alpha=0.7, linestyle="-."
)
(cre_interp_mean["jed0022"]["lw"] - cre_interp_mean["jed0011"]["lw"]).plot(
    ax=axes[0], color="r", alpha=0.7, linestyle="--"
)
(cre_interp_mean["jed0033"]["sw"] - cre_interp_mean["jed0011"]["sw"]).plot(
    ax=axes[1], color="blue", alpha=0.7, linestyle="-."
)
(cre_interp_mean["jed0022"]["sw"] - cre_interp_mean["jed0011"]["sw"]).plot(
    ax=axes[1], color="blue", alpha=0.7, linestyle="--"
)
(cre_interp_mean["jed0033"]["net"] - cre_interp_mean["jed0011"]["net"]).plot(
    ax=axes[2], color="k", alpha=0.7, linestyle="-."
)
(cre_interp_mean["jed0022"]["net"] - cre_interp_mean["jed0011"]["net"]).plot(
    ax=axes[2], color="k", alpha=0.7, linestyle="--"
)

labels = ["+2K - control", "+4K - control"]
handles = [
    plt.Line2D([0], [0], color="grey", linestyle="-."),
    plt.Line2D([0], [0], color="grey", linestyle="--"),
]
fig.legend(handles, labels, bbox_to_anchor=[0.7, 0], frameon=True, ncols=2)

for ax in axes:
    ax.set_xlim([1e-4, 4e1])
    ax.set_ylim([-5, 15])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("$I$  / kg m$^{-2}$")
    ax.set_xscale("log")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)

axes[0].set_ylabel(r"$\Delta C_{\mathrm{lw}}(I)$  / W m$^{-2}$")
axes[1].set_ylabel(r"$\Delta C_{\mathrm{sw}}(I)$  / W m$^{-2}$")
axes[2].set_ylabel(r"$\Delta C_{\mathrm{net}}(I)$  / W m$^{-2}$")
fig.savefig("plots/cre_iwp_diff.png", dpi=300, bbox_inches="tight")

# %% read cloudsat
cloudsat = read_cloudsat("2009")

# %% calculate iwp hist
iwp_bins = np.logspace(-5, np.log10(40), 51)
histograms = {}
for run in runs:
    iwp = datasets[run]["clivi"] + datasets[run]["qsvi"] + datasets[run]["qgvi"]
    histograms[run], edges = np.histogram(iwp, bins=iwp_bins, density=False)
    histograms[run] = histograms[run] / len(iwp)

histograms["cloudsat"], _ = np.histogram(cloudsat["ice_water_path"]/1e3, bins=iwp_bins, density=False)
histograms["cloudsat"] = histograms["cloudsat"] / len(cloudsat)

# %% plot iwp hists
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
colors = {"jed0011": "k", "jed0022": "red", "jed0033": "orange"}
ax.stairs(histograms["cloudsat"], edges, label="CloudSat", color="k", linewidth=4, alpha=0.5)
for run in runs:
    ax.stairs(histograms[run], edges, label=exp_name[run], color=colors[run])

ax.legend()
ax.set_xscale("log")
ax.set_ylabel("P(IWP)")
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_xlim([1e-4, 40])
ax.spines[["top", "right"]].set_visible(False)
fig.savefig("plots/iwp_hist_rand.png", dpi=300, bbox_inches="tight")

# %% plot diff to control 
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
for run in ["jed0022", "jed0033"]:
    diff = histograms[run] - histograms["jed0011"]
    ax.stairs(diff, edges, label=exp_name[run], color=colors[run])

ax.legend()
ax.set_xscale("log")
ax.set_ylabel("P(IWP)")
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_xlim([1e-4, 40])
ax.spines[["top", "right"]].set_visible(False)
fig.savefig("plots/iwp_hist_rand_diff.png", dpi=300, bbox_inches="tight")

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
ax.set_xscale('log')
ax.set_xlim([1e-4, 40])
fig.savefig("plots/cre_iwp_folded.png", dpi=300, bbox_inches="tight")

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


feedback['jed0033'] = (cre_integrated['jed0033'] - cre_integrated["jed0011"]) / 2
feedback['jed0022'] = (cre_integrated['jed0022'] - cre_integrated["jed0011"]) / 4
feedback_const_iwp['jed0033'] = (cre_const_iwp_integrated['jed0033'] - cre_const_iwp_integrated["jed0011"]) / 2
feedback_const_iwp['jed0022'] = (cre_const_iwp_integrated['jed0022'] - cre_const_iwp_integrated["jed0011"]) / 4
feedback_const_cre['jed0033'] = (cre_const_cre_integrated['jed0033'] - cre_const_cre_integrated["jed0011"]) / 2
feedback_const_cre['jed0022'] = (cre_const_cre_integrated['jed0022'] - cre_const_cre_integrated["jed0011"]) / 4

# %% plot integrated CRE and feedback 
fig, axes = plt.subplots(1, 3, figsize=(9, 4), sharex=True, sharey=True)


axes[0].scatter(0, feedback["jed0033"]['lw'].values, color='red', marker='x')
axes[0].scatter(1, feedback["jed0033"]['sw'].values, color='blue', marker='x')
axes[0].scatter(2, feedback["jed0033"]['net'].values, color='k', marker='x')

axes[0].scatter(0, feedback["jed0022"]['lw'].values, color='red', marker='o', facecolors='none')
axes[0].scatter(1, feedback["jed0022"]['sw'].values, color='blue', marker='o', facecolors='none')
axes[0].scatter(2, feedback["jed0022"]['net'].values, color='k', marker='o', facecolors='none')
axes[0].set_title("Total")

axes[1].scatter(0, feedback_const_iwp["jed0033"]['lw'].values, color='red', marker='x')
axes[1].scatter(1, feedback_const_iwp["jed0033"]['sw'].values, color='blue', marker='x')
axes[1].scatter(2, feedback_const_iwp["jed0033"]['net'].values, color='k', marker='x')

axes[1].scatter(0, feedback_const_iwp["jed0022"]['lw'].values, color='red', marker='o', facecolors='none')
axes[1].scatter(1, feedback_const_iwp["jed0022"]['sw'].values, color='blue', marker='o', facecolors='none')
axes[1].scatter(2, feedback_const_iwp["jed0022"]['net'].values, color='k', marker='o', facecolors='none')
axes[1].set_title("CRE Change")

axes[2].scatter(0, feedback_const_cre["jed0033"]['lw'].values, color='red', marker='x')
axes[2].scatter(1, feedback_const_cre["jed0033"]['sw'].values, color='blue', marker='x')
axes[2].scatter(2, feedback_const_cre["jed0033"]['net'].values, color='k', marker='x')

axes[2].scatter(0, feedback_const_cre["jed0022"]['lw'].values, color='red', marker='o', facecolors='none')
axes[2].scatter(1, feedback_const_cre["jed0022"]['sw'].values, color='blue', marker='o', facecolors='none')
axes[2].scatter(2, feedback_const_cre["jed0022"]['net'].values, color='k', marker='o', facecolors='none')
axes[2].set_title("IWP Change")

for ax in axes:
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([-0.5, 0, 0.5, 1])
    ax.set_xticklabels(['LW', 'SW', 'Net'])
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)


axes[0].set_ylabel('Feedback / W m$^{-2}$ K$^{-1}$')

labels = ["+2K", "+4K"]
handles = [
    plt.Line2D([0], [0], color="grey", marker='x', linestyle="none"),
    plt.Line2D([0], [0], color="grey", marker='o', linestyle='none'),
]
fig.legend(handles, labels, bbox_to_anchor=[0.62, 0], frameon=True, ncols=2)
fig.savefig("plots/feedback.png", dpi=300, bbox_inches="tight")


# %% investigate low cloud fraction
fig, ax = plt.subplots()
datasets["jed0011"]["mask_low_cloud"].groupby_bins(
    datasets["jed0011"]["iwp"], iwp_bins
).mean().plot(ax=ax, label="control", color="k")
datasets["jed0033"]["mask_low_cloud"].groupby_bins(
    datasets["jed0033"]["iwp"], iwp_bins
).mean().plot(ax=ax, label="+2K", color="orange")
datasets["jed0022"]["mask_low_cloud"].groupby_bins(
    datasets["jed0022"]["iwp"], iwp_bins
).mean().plot(ax=ax, label="+4K", color="red")

ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("$I$  / kg m$^{-2}$")
ax.set_ylabel(r"$f_{\mathrm{lc}}(I)$")
ax.set_xscale("log")
ax.set_xlim([1e-4, 4e1])
ax.legend()
fig.savefig('plots/low_cloud_fraction.png', dpi=300, bbox_inches='tight')

# %% investigate high cloud temperature and pressure
fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

datasets["jed0011"]["hc_top_temperature"].groupby_bins(
    datasets["jed0011"]["iwp"], iwp_bins
).mean().plot(ax=axes[0], label="control", color="k")
datasets["jed0033"]["hc_top_temperature"].groupby_bins(
    datasets["jed0033"]["iwp"], iwp_bins
).mean().plot(ax=axes[0], label="+2K", color="orange")
datasets["jed0022"]["hc_top_temperature"].groupby_bins(
    datasets["jed0022"]["iwp"], iwp_bins
).mean().plot(ax=axes[0], label="+4K", color="red")

datasets["jed0011"]["hc_top_pressure"].groupby_bins(
    datasets["jed0011"]["iwp"], iwp_bins
).mean().plot(ax=axes[1], label="control", color="k")
datasets["jed0033"]["hc_top_pressure"].groupby_bins(
    datasets["jed0033"]["iwp"], iwp_bins
).mean().plot(ax=axes[1], label="+2K", color="orange")
datasets["jed0022"]["hc_top_pressure"].groupby_bins(
    datasets["jed0022"]["iwp"], iwp_bins
).mean().plot(ax=axes[1], label="+4K", color="red")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim([1e-4, 4e1])


axes[0].set_ylabel(r"$T_{\mathrm{hc}}$  / K")
axes[1].set_ylabel(r"$p_{\mathrm{hc}}$  / hPa")
axes[1].set_xlabel("$I$  / kg m$^{-2}$")
axes[0].set_xlabel("")
axes[0].set_xscale("log")

axes[0].legend()
fig.savefig("plots/hc_top.png", dpi=300, bbox_inches="tight")

# %%
