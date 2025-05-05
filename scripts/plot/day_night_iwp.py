# %%
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

# %% load data
runs = ["jed0011", "jed0022", "jed0033"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
colors = {"jed0011": "k", "jed0022": "red", "jed0033": "orange"}
datasets = {}
cre_interp_mean = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed.nc"
    )
    cre_interp_mean[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/cre/{run}_cre_interp_mean_rand.nc"
    )

# %% calculate masks
mode = "const_lc"
masks_height = {}
for run in runs:
    if mode == "pressure":
        masks_height[run] = datasets[run]["hc_top_pressure"] < 350
    elif mode == "raw":
        masks_height[run] = True
    else:
        masks_height[run] = datasets[run]["hc_top_temperature"] < (273.15 - 35)

# %% calculate iwp hist
iwp_bins = np.logspace(-4, np.log10(40), 51)
histograms_day = {}  
histograms_night = {}
histograms_morning = {}
histograms_noon = {}
histograms_evening = {}
histograms_midnight = {}

for run in runs:
    iwp_day = datasets[run]["iwp"].where(masks_height[run] & ((datasets[run]['time_local'] > 6) & (datasets[run]['time_local'] < 18)))
    histograms_day[run], edges = np.histogram(iwp_day, bins=iwp_bins, density=False)
    histograms_day[run] = histograms_day[run] / len(iwp_day)
    iwp_night = datasets[run]["iwp"].where(masks_height[run] & ((datasets[run]['time_local'] < 6) | (datasets[run]['time_local'] > 18)))
    histograms_night[run], edges = np.histogram(iwp_night, bins=iwp_bins, density=False)
    histograms_night[run] = histograms_night[run] / len(iwp_night)
    iwp_morning = datasets[run]["iwp"].where(masks_height[run] & (datasets[run]['time_local'] > 3) & (datasets[run]['time_local'] < 9))
    histograms_morning[run], edges = np.histogram(iwp_morning, bins=iwp_bins, density=False)
    histograms_morning[run] = histograms_morning[run] / len(iwp_morning)
    iwp_noon = datasets[run]["iwp"].where(masks_height[run] & (datasets[run]['time_local'] > 9) & (datasets[run]['time_local'] < 15))
    histograms_noon[run], edges = np.histogram(iwp_noon, bins=iwp_bins, density=False)
    histograms_noon[run] = histograms_noon[run] / len(iwp_noon)
    iwp_evening = datasets[run]["iwp"].where(masks_height[run] & (datasets[run]['time_local'] > 15) & (datasets[run]['time_local'] < 21))
    histograms_evening[run], edges = np.histogram(iwp_evening, bins=iwp_bins, density=False)
    histograms_evening[run] = histograms_evening[run] / len(iwp_evening)
    iwp_midnight = datasets[run]["iwp"].where(masks_height[run] & (datasets[run]['time_local'] > 21) | (datasets[run]['time_local'] < 3))
    histograms_midnight[run], edges = np.histogram(iwp_midnight, bins=iwp_bins, density=False)
    histograms_midnight[run] = histograms_midnight[run] / len(iwp_midnight)

# %% plot iwp hists day night
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for run in ['jed0033', 'jed0022']:
    axes[0].stairs(histograms_day[run] - histograms_day['jed0011'], edges, label=exp_name[run], color=colors[run])
    axes[1].stairs(histograms_night[run] - histograms_night['jed0011'], edges, label=exp_name[run], color=colors[run])
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("$I$  / kg m$^{-2}$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim([1e-4, 4e1])
    ax.axhline(0, color="black", lw=0.5)
axes[0].set_ylabel("$P(I)$")
axes[0].set_title("Day")
axes[1].set_title("Night")
axes[1].legend()
fig.savefig('plots/variability/iwp_day_night.png', dpi=300, bbox_inches='tight')
# %% plot iwp hists time of day 
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
for run in ['jed0033', 'jed0022']:
    axes[0].stairs(histograms_morning[run] - histograms_morning['jed0011'], edges, label=exp_name[run], color=colors[run])
    axes[1].stairs(histograms_noon[run] - histograms_noon['jed0011'], edges, label=exp_name[run], color=colors[run])
    axes[2].stairs(histograms_evening[run] - histograms_evening['jed0011'], edges, label=exp_name[run], color=colors[run])
    axes[3].stairs(histograms_midnight[run] - histograms_midnight['jed0011'], edges, label=exp_name[run], color=colors[run])
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("$I$  / kg m$^{-2}$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim([1e-4, 4e1])
    ax.axhline(0, color="black", lw=0.5)
axes[0].set_ylabel("$P(I)$")
axes[0].set_title("Morning")
axes[1].set_title("Noon")
axes[2].set_title("Evening")
axes[3].set_title("Midnight")
axes[3].legend()
fig.savefig('plots/variability/iwp_time_of_day.png', dpi=300, bbox_inches='tight')

# %% plot iwp hists south north 
hists_north = {}
hists_south = {}
for run in runs:
    iwp_north = datasets[run]["iwp"].where( (datasets[run]['clat'] > 0))
    hists_north[run], edges = np.histogram(iwp_north, bins=iwp_bins, density=False)
    hists_north[run] = hists_north[run] / len(iwp_north)
    iwp_south = datasets[run]["iwp"].where((datasets[run]['clat'] < 0))
    hists_south[run], edges = np.histogram(iwp_south, bins=iwp_bins, density=False)
    hists_south[run] = hists_south[run] / len(iwp_south)


# %% plot iwp hists north south
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for run in ['jed0033', 'jed0022']:
    axes[0].stairs(hists_north[run] - hists_north['jed0011'], edges, label=exp_name[run], color=colors[run])
    axes[1].stairs(hists_south[run] - hists_south['jed0011'], edges, label=exp_name[run], color=colors[run])
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("$I$  / kg m$^{-2}$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim([1e-4, 4e1])
    ax.axhline(0, color="black", lw=0.5)
axes[0].set_title("North")
axes[1].set_title("South")
fig.savefig('plots/variability/iwp_north_south.png', dpi=300, bbox_inches='tight')

# %%
