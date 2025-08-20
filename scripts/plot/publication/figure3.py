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
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/deep_clouds_daily_cycle_01.nc"
    )
hists_5 = {}
for run in runs:
    hists_5[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/deep_clouds_daily_cycle_5.nc"
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
edges = np.arange(0, 25, 1)
for run in runs:
    histograms_iwp[run] = (hists[run].sum("day") / hists['jed0011'].sum())[
        "__xarray_dataarray_variable__"
    ].values
    histograms_iwp_5[run] = (hists_5[run].sum("day") / hists_5[run].sum())[
        "__xarray_dataarray_variable__"
    ].values

# %% plot
fig, ax1 = plt.subplots(figsize=(8, 4))



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
ax1.set_ylim([0.03, 0.06])
ax1.set_yticks([0.03, 0.04, 0.05])
ax2.set_ylim([0, 1400])
ax2.set_ylabel("Incoming SW Radiation / W m$^{-2}$", color='grey')
ax2.set_yticks([0, 700, 1400])
ax2.tick_params(axis='y', labelcolor='grey')
# add legend
ax1.legend(frameon=False)

#fig.savefig("plots/publication/figure3_alt.pdf", bbox_inches="tight")

# %% calculate time difference to noon 
mean_rad_time = {}
for run in runs:
    rad_time = np.abs(datasets[run]['time_local'] - 12)
    mean_rad_time[run] = rad_time.where(datasets[run]['iwp'] > 1).mean()
    print(f"{exp_name[run]}: {mean_rad_time[run].values:.2f} h")
for run in runs[1:]:
    print(f"Time Difference Time to Nooon for {exp_name[run]}: {((mean_rad_time[run].values - mean_rad_time['jed0011'].values)*60):.2f} min")

# %% calculate change in time needed for feedback 
feedbacks = {
    'jed0022': 0.16,
    'jed0033': 0.31,
}
feedback_lw = 0.33

time_diff_feedback = {}

for run in runs[1:]:
    time_diff_feedback[run] = ((mean_rad_time[run] - mean_rad_time['jed0011']) / T_delta[run]) * (feedback_lw / feedbacks[run])
    print(f"Time Difference for {exp_name[run]}: {time_diff_feedback[run].values * 60:.0f} min / K")

# %% calculate difference in incoming SW radiation 
sw_in = {}
for run in runs:
    sw_in[run] = datasets[run]['rsdt'].where(datasets[run]['iwp'] > 1).mean()
    print(f"{exp_name[run]}: {sw_in[run].values:.2f} W m$^{-2}$")

for run in runs[1:]:
    print(f"Difference in Incoming SW Radiation for {exp_name[run]}: {sw_in[run].values - sw_in['jed0011'].values:.2f} W m$^{-2}$")

# %% calculate time difference to noon from histograms
edge_centers = (edges[:-1] + edges[1:]) / 2
mean_rad_time_hist = {}
mean_rad_hist = {}
SW_in = (
    datasets["jed0011"]["rsdt"]
    .groupby_bins(datasets["jed0011"]["time_local"], bins=edges)
    .mean()
)
# %%
for run in runs:
    weighted_time = np.abs(edge_centers - 12) * histograms_iwp[run]
    mean_rad_time_hist[run] = np.sum(weighted_time)
    weighted_SW = SW_in * histograms_iwp[run]
    mean_rad_hist[run] = np.sum(weighted_SW)

    print(f"{exp_name[run]}: {mean_rad_time_hist[run]:.2f} h")
    print(f"{exp_name[run]}: {mean_rad_hist[run]:.2f} W m$^{-2}$") 

for run in runs[1:]:
    print(f"Time Difference Time to Nooon for {exp_name[run]}: {((mean_rad_time_hist[run] - mean_rad_time_hist['jed0011'])*60):.2f} min")
    print(f"Difference in Incoming SW Radiation for {exp_name[run]}: {mean_rad_hist[run] - mean_rad_hist['jed0011']:.2f} W m$^{-2}$")

# %%
