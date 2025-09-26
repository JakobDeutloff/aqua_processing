# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from src.read_data import load_random_datasets, load_definitions

# %% load CRE data
runs, exp_name, colors, line_labels, sw_color, lw_color, net_color, linestyles = (
    load_definitions()
)
iwp_bins = np.logspace(-4, np.log10(40), 51)
datasets = load_random_datasets()
T_delta = {
    "jed0011": 0,
    "jed0022": 4,
    "jed0033": 2,
}
hists = {}
for run in runs:
    hists[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/deep_clouds_daily_cycle.nc"
    )
hists_5 = {}
for run in runs:
    hists_5[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/deep_clouds_daily_cycle_5.nc"
    )
hists_01 = {}
for run in runs:
    hists_01[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/deep_clouds_daily_cycle_01.nc"
    )

# %% get normalised incoming SW for every bin
bins_gradient = np.arange(0, 24.01, 0.1)
SW_in = (
    datasets["jed0011"]["rsdt"]
    .groupby_bins(datasets["jed0011"]["time_local"], bins=bins_gradient)
    .mean()
)

# %% calculate histograms
histograms_iwp = {}
histograms_iwp_5 = {}
histograms_iwp_01 = {}
edges = np.arange(0, 25, 1)
for run in runs:
    histograms_iwp[run] = (hists[run].sum("day") / hists['jed0011'].sum())[
        "__xarray_dataarray_variable__"
    ].values
    histograms_iwp_5[run] = (hists_5[run].sum("day") / hists_5['jed0011'].sum())[
        "__xarray_dataarray_variable__"
    ].values
    histograms_iwp_01[run] = (hists_01[run].sum("day") / hists_01['jed0011'].sum())[
        "__xarray_dataarray_variable__"
    ].values

# %% plot 1 kg m^-2
fig, ax1 = plt.subplots(figsize=(5, 2.5))

for run in runs:
    ax1.stairs(
        histograms_iwp[run], edges, label=line_labels[run], color=colors[run],
    )
ax2 = ax1.twinx()
SW_in.plot(
    ax=ax2,
    color="grey",
    linewidth=3,
    alpha=0.5)

for ax in [ax1, ax2]:
    ax.set_xlim([0.1, 23.9])
    ax.spines[["top"]].set_visible(False)
    ax.set_xticks([6, 12, 18])
    ax.set_xlabel("Local Time / h")

ax1.set_ylabel("P($I$ > 1 kg m$^{-2}$)")
ax1.set_ylim([0.03, 0.051])
ax1.set_yticks([0.03, 0.04, 0.05])
ax2.set_ylim([0, 1400])
ax2.set_ylabel("Incoming SW Radiation / W m$^{-2}$", color='grey')
ax2.set_yticks([0, 700, 1400])
ax2.tick_params(axis='y', labelcolor='grey')
# add legend
ax1.legend(frameon=False)

fig.savefig("plots/publication/diurnal_cycle_1.pdf", bbox_inches="tight")

# %% plot 5 kg m^-2
fig, ax1 = plt.subplots(figsize=(8, 4))

for run in runs:
    ax1.stairs(
        histograms_iwp_5[run], edges, label=line_labels[run], color=colors[run],
    )
ax2 = ax1.twinx()
SW_in.plot(
    ax=ax2,
    color="grey",
    linewidth=3,
    alpha=0.5)

for ax in [ax1, ax2]:
    ax.set_xlim([0.1, 23.9])
    ax.spines[["top"]].set_visible(False)
    ax.set_xticks([6, 12, 18])
    ax.set_xlabel("Local Time / h")

ax1.set_ylabel("P($I$ > 5 kg m$^{-2}$)")
ax1.set_ylim([0.03, 0.057])
ax1.set_yticks([0.03, 0.04, 0.05])
ax2.set_ylim([0, 1400])
ax2.set_ylabel("Incoming SW Radiation / W m$^{-2}$", color='grey')
ax2.set_yticks([0, 700, 1400])
ax2.tick_params(axis='y', labelcolor='grey')
# add legend
ax1.legend(frameon=False)

fig.savefig("plots/publication/sup_5kg_dc.pdf", bbox_inches="tight")

# %% plot 0.1 kg m^-2
fig, ax1 = plt.subplots(figsize=(8, 4))

for run in runs:
    ax1.stairs(
        histograms_iwp_01[run], edges, label=line_labels[run], color=colors[run],
    )
ax2 = ax1.twinx()
SW_in.plot(
    ax=ax2,
    color="grey",
    linewidth=3,
    alpha=0.5)

for ax in [ax1, ax2]:
    ax.set_xlim([0.1, 23.9])
    ax.spines[["top"]].set_visible(False)
    ax.set_xticks([6, 12, 18])
    ax.set_xlabel("Local Time / h")

ax1.set_ylabel("P($I$ > 10$^{-1}$ kg m$^{-2}$)")
ax1.set_ylim([0.03, 0.057])
ax1.set_yticks([0.03, 0.04, 0.05])
ax2.set_ylim([0, 1400])
ax2.set_ylabel("Incoming SW Radiation / W m$^{-2}$", color='grey')
ax2.set_yticks([0, 700, 1400])
ax2.tick_params(axis='y', labelcolor='grey')
# add legend
ax1.legend(frameon=False)

fig.savefig("plots/publication/sup_01kg_dc.pdf", bbox_inches="tight")

# %% calculate difference between peaks 
diffs = {}
temp_delats = {
    'jed0022': 4,
    'jed0033': 2,
}
for run in runs:
    max_morning = histograms_iwp[run].max()
    max_aternoon = histograms_iwp[run][12:20].max()
    diffs[run] = (max_morning - max_aternoon)

for run in runs[1:]:
    print(f"Increase in diurnal cycle for {line_labels[run]} compared to control by {(((diffs[run]-diffs['jed0011']) / diffs['jed0011'])*100):.0f}%")

# %%
