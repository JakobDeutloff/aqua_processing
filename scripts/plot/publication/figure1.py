# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from src.read_data import read_cloudsat, load_iwp_hists, load_cre, load_random_datasets

# %% load data
runs = ["jed0011", "jed0033", "jed0022"]
colors = {"jed0011": "#3e0237", "jed0022": "#f707da", "jed0033": "#9a0488"}
linestyles = {
    "jed0011": "-",
    "jed0022": "--",
    "jed0033": "-.",
}
line_labels = {
    "jed0011": "Control",
    "jed0022": "+4 K",
    "jed0033": "+2 K",
}
mode = "raw"
datasets = load_random_datasets(processed=True)
histograms = load_iwp_hists()
cre = load_cre()
# %% read cloudsat
cloudsat_raw = read_cloudsat("2009")

# %% average over pairs of three entries in cloudsat to get to a resolution of 4.8 km
cloudsat = cloudsat_raw.to_xarray().coarsen({"scnline": 3}, boundary="trim").mean()

# %% calculate iwp hist
iwp_bins = np.logspace(-4, np.log10(40), 51)
histograms["cloudsat"], edges = np.histogram(
    cloudsat["ice_water_path"] / 1e3, bins=iwp_bins, density=False
)
histograms["cloudsat"] = histograms["cloudsat"] / len(cloudsat["ice_water_path"])

# %% plot
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, figure=fig)

# Create the main 2x2 axes
ax00 = fig.add_subplot(gs[0, 0])
ax11 = fig.add_subplot(gs[1, 1])
ax01 = fig.add_subplot(gs[0, 1])

# Now split axes[1, 1] into two (vertical split)
gs_sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0])
ax10 = fig.add_subplot(gs_sub[0, 0])
ax20 = fig.add_subplot(gs_sub[1, 0])
axes = [ax00, ax11, ax01, ax10, ax20]

# CRE
ax00.axhline(0, color="k", linewidth=0.5)
ax10.axhline(0, color="k", linewidth=0.5)
ax20.axhline(0, color="k", linewidth=0.5)
for run in runs:
    ax00.plot(cre[run]["iwp"], cre[run]["lw"], color="r", linestyle=linestyles[run])
    ax00.plot(cre[run]["iwp"], cre[run]["sw"], color="b", linestyle=linestyles[run])
    ax00.plot(cre[run]["iwp"], cre[run]["net"], color="k", linestyle=linestyles[run])
ax00.set_yticks([-250, 0, 200])
ax00.set_ylabel(r"$C(I)$ / W m$^{-2}$")

for run in runs[1:]:
    ax10.plot(cre[run]["iwp"], cre[run]["lw"] - cre[runs[0]]["lw"], color=colors[run], label=line_labels[run])
    ax20.plot(cre[run]["iwp"], cre[run]["sw"] - cre[runs[0]]["sw"], color=colors[run], label=line_labels[run])
ax10.set_ylim([-1, 28])
ax10.set_ylabel(r"$\Delta C_{\mathrm{LW}}(I)$ / W m$^{-2}$", color="r")
ax10.set_yticks([0, 5, 20])
ax20.set_ylim([-1, 28])
ax20.set_ylabel(r"$\Delta C_{\mathrm{SW}}(I)$ / W m$^{-2}$", color="b")
ax20.set_yticks([0, 5, 20])

# IWP histograms
ax01.stairs(histograms["cloudsat"], edges, color="k", linewidth=4, alpha=0.5, label="2C-ICE")
for run in runs:
    ax01.stairs(
        histograms[run],
        edges,
        color=colors[run],
        label=line_labels[run],
        linewidth=1.5
    )
ax01.set_yticks([0, 0.01, 0.02])
ax01.set_ylabel("$P(I)$")
ax11.axhline(0, color="k", linewidth=0.5)
for run in runs[1:]:
    ax11.stairs(
        histograms[run] - histograms[runs[0]],
        edges,
        color=colors[run],
        label=line_labels[run],
        linewidth=1.5,
    )
ax11.set_yticks([-0.0004, 0, 0.0008])
ax11.set_ylabel(r"$\Delta P(I)$")

for ax in [ax00, ax10, ax20, ax01, ax11]:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 20)

for ax in [ax00, ax10, ax01]:
    ax.set_xticklabels([])

for ax in [ax20, ax11]:
    ax.set_xlabel("$I$ / kg m$^{-2}$")

# legends 
for ax in [ax10, ax20]:
    ax.legend(frameon=False, loc="upper left", fontsize=10)
for ax in [ax01, ax11]:
    ax.legend(frameon=False, loc="upper right", fontsize=10)
handles = [plt.Line2D([0], [0], color='r'), 
           plt.Line2D([0], [0], color='b'), 
           plt.Line2D([0], [0], color='k'),
           plt.Line2D([0], [0], color='grey'),
           plt.Line2D([0], [0], color='grey', linestyle='--'),
           plt.Line2D([0], [0], color='grey', linestyle='-.')]
labels = ['LW', 'SW', 'Net', '2C-ICE', 'Control', '+4 K', '+2 K'] 
ax00.legend(handles=handles, labels=labels, loc='lower left', frameon=False, fontsize=10, ncols=2)    

# add letters 
for ax, letter in zip([ax00, ax10, ax20, ax01, ax11], ["a", "b", "c", "d", "e"]):
    ax.text(0.03, 1, letter, transform=ax.transAxes, fontsize=14, fontweight='bold')

fig.tight_layout()
fig.savefig('plots/publication/figure1.pdf', bbox_inches='tight')

# %%
