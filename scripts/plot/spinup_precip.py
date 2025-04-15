# %%
import xarray as xr
import matplotlib.pyplot as plt
import easygems.healpix as egh
import numpy as np
import matplotlib as mpl
import pandas as pd

# %%
runs = ["jed0011", "jed0033", "jed0022"]
spinups = {"jed0011": "jed0001", "jed0022": "jed0002", "jed0033": "jed0003"}
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
colors = {"jed0011": "k", "jed0022": "r", "jed0033": "orange"}
datasets = {}
# %%
for run in runs:
    print(run)
    ds_spinup = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/spinup/{spinups[run]}_atm_2d_daymean_hp.nc"
    ).pipe(egh.attach_coords)['pr']
    ds_prod = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/{run}_atm_2d_daymean_hp.nc"
    ).pipe(egh.attach_coords)['pr']
    time = pd.date_range(ds_spinup['time'][-1].values, freq='d', periods=len(ds_prod['day']))
    ds_prod['day'] = time
    ds_prod = ds_prod.rename({"day": "time"})
    datasets[run] = xr.concat([ds_spinup.isel(time=slice(None, -1)), ds_prod], dim="time")

# %% calculate zonal mean precip 
lat_bins = np.linspace(-20, 20, 100)
lat_points = (lat_bins[:-1] + lat_bins[1:]) / 2
precip_zonal = {}
for run in runs:
    print(run)
    precip_zonal[run] = datasets[run].groupby_bins(datasets[run]["lat"], lat_bins).mean()

# %%calculate mean and std of zonal precip
pr_mean = {}
pr_std = {}
for run in runs:
    pr_mean[run] = (precip_zonal[run]*60*60*24).isel(time=slice(-60, None)).mean(dim="time")
    pr_std[run] = (precip_zonal[run]*60*60*24).isel(time=slice(-60, None)).std(dim="time")

# %%
fig, ax = plt.subplots()
for run in ['jed0022', 'jed0033', 'jed0011']:
    ax.plot(lat_points, pr_mean[run], label=exp_name[run], color=colors[run])
    ax.fill_between(lat_points, pr_mean[run] - pr_std[run], pr_mean[run] + pr_std[run], alpha=0.2, color=colors[run])

ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel("Latitude")
ax.set_ylabel("Precipitation / mm day$^{-1}$")
ax.set_xlim([-30, 30])

handles, labels = ax.get_legend_handles_labels()
handles.append(mpl.patches.Patch(color='grey', alpha=0.2))
labels.append(r'$\pm  \sigma$')
ax.legend(handles=handles, labels=labels)

# %% calculate asymmetrie between peaks in precip 
assym = {}
for run in runs:
    peak_north = precip_zonal[run].sel(lat_bins=precip_zonal[run].lat_bins[lat_points > 0]).max(dim="lat_bins")*60*60*24
    peak_south = precip_zonal[run].sel(lat_bins=precip_zonal[run].lat_bins[lat_points < 0]).max(dim="lat_bins")*60*60*24
    assym[run] = (peak_north - peak_south) 

# %% plot assym 
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharey=True)
for i, run in enumerate(runs):
    axes[i].axhline(0, color="k", linestyle="-", linewidth=0.5)
    axes[i].plot(assym[run], label=run, color=colors[run])
    axes[i].set_xlabel("Day")
    axes[i].set_ylabel(r"$P_{max, N}$ - $P_{max, S}$  / mm day${-1}$")
    axes[i].spines[["top", "right"]].set_visible(False)
    axes[i].text(
        0.3,
        0.9,
        f"Mean Asymmetrie: {assym[run].mean().values:.2f} mm day$^{-1}$",
        transform=axes[i].transAxes,
        ha="center",
        va="center",
        fontsize=12,
    )
    axes[i].axvline(len(assym[run]['time'])-30, color="grey", linestyle="--")
fig.savefig('plots/spinup/asymmetrie.png', dpi=300, bbox_inches='tight')

# %% plot symmetrical and asymmetrical component of control spinup 


# %%
