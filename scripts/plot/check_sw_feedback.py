# %% 
import matplotlib.pyplot as plt
import xarray as xr
from src.read_data import load_random_datasets, load_definitions
import numpy as np

# %%
runs, exp_name, colors, line_labels, sw_color, lw_color, net_color, linestyles = (
    load_definitions()
)
datasets = load_random_datasets()

# %% 
s_hc = {}
s_lc = {}
for run in ['jed0011', 'jed0033', 'jed0022']:
    s_hc[run] = datasets[run]['rsdt'].where(datasets[run]['iwp']>1).mean()
    s_lc[run] = datasets[run]['rsdt'].where((datasets[run]['iwp']>1) & (datasets[run]['lwp']>1e-3)).mean()
    print(f"{run}: high {s_hc[run].values:.2f} W/m^2, low {s_lc[run].values:.2f} W/m^2")

# %% incoming sloar as function of IWP 
s_iwp = {}
for run in runs:
    s_iwp[run] = datasets[run]['rsdt'].groupby_bins(datasets[run]['iwp'], bins=np.logspace(-4, np.log10(40), 51)).mean()

# %% plot 
fig, ax = plt.subplots()
for run in runs:
    ax.plot(np.logspace(-4, np.log10(40), 50), s_iwp[run].values, label=line_labels[run], color=colors[run], linestyle=linestyles[run])
ax.set_xscale('log')

# %% load hists 
hists={}
hists_coarse={}
for run in runs:
    hists[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/daily_cycle_hist_2d.nc"
    ).coarsen(iwp=4, boundary="trim").sum()

    hists_coarse[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/daily_cycle_hist_2d_0utc.nc"
    ).coarsen(iwp=4, boundary="trim").sum()


# %% calculate received SW from histograms 
S_in = datasets['jed0011']['rsdt'].groupby_bins(datasets['jed0011']['time_local'], bins=np.arange(0, 25, 1)).mean()
S_in = S_in.rename({'time_local_bins': 'local_time'})
S_in['local_time'] = hists['jed0011']['local_time']  # align local time coordinates
#%% calculate mean of S_in weighted with hist binned by iwp
s_hist = {}
s_hist_coarse = {}
for run in runs:
    hist_norm = hists[run]['hist'] * S_in
    s_hist[run] = hist_norm.sum(dim=['local_time', 'time']) / hists[run]['hist'].sum(dim=['local_time', 'time'])
    hist_norm_coarse = hists_coarse[run]['hist'] * S_in
    s_hist_coarse[run] = hist_norm_coarse.sum(dim=['local_time', 'time']) / hists_coarse[run]['hist'].sum(dim=['local_time', 'time'])


# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True, sharex=True)
for run in runs:
    axes[1].plot(hists_coarse[run]['iwp'], s_hist_coarse[run].values, color=colors[run])
    axes[2].plot(hists[run]['iwp'], s_hist[run].values, color=colors[run])
    axes[0].plot(np.logspace(-4, np.log10(40), 50), s_iwp[run].values, color=colors[run])
for ax in axes:
    ax.set_xscale('log')
    ax.set_xlabel('IWP (g/m^2)')
    ax.set_ylim(350, 475)
    ax.set_xlim(1e-3, 10)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('Incoming SW (W/m^2)')
# %% load full iwp and full 2d rsdt 
iwps = {}
rsdts = {}
for run in ['jed0022']:
    iwps[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/{run}_iwp.nc"
    )



# %% calculate time distance to noon from histograms
weighted_time_dist = {}
for run in runs:
    time_distance_noon = np.abs(hists[run]['local_time'] - 12)
    weighted_time_dist[run] = (hists[run]['hist'] * time_distance_noon).sum(dim=['local_time', 'time']) / hists[run]['hist'].sum(dim=['local_time', 'time'])
# %%
fig, ax = plt.subplots()
for run in runs:
    ax.plot(hists[run]['iwp'], weighted_time_dist[run].values, label=line_labels[run], color=colors[run])
ax.set_xscale('log')
ax.invert_yaxis()

# %% plot histograms for +4K at low I
hist_4k_hist = hists['jed0022'].sel(iwp=slice(1e-3, 1e-2))['hist'].sum(['time', 'iwp']) / hists['jed0022'].sel(iwp=slice(1e-3, 1e-2))['hist'].sum()
hist_4K_sample = np.histogram(datasets['jed0022'].where((datasets['jed0022']['iwp'] < 1e-2) & (datasets['jed0022']['iwp'] > 1e-3))['local_time'].values, bins=np.arange(0, 24.1, 0.1))
