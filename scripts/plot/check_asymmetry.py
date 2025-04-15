# %%
import xarray as xr
import matplotlib.pyplot as plt
import easygems.healpix as egh
import numpy as np
from scipy.stats import wasserstein_distance
import pandas as pd

# %%
ds_spinup = xr.open_dataset(
    f"/work/bm1183/m301049/icon_hcap_data/control/spinup/jed0001_atm_2d_daymean_hp.nc"
).pipe(egh.attach_coords)[['clivi', 'pr']]
ds_prod = xr.open_dataset(
    f"/work/bm1183/m301049/icon_hcap_data/control/production/jed0011_atm_2d_daymean_hp.nc"
).pipe(egh.attach_coords)[['clivi', 'pr']]
time = pd.date_range(ds_spinup['time'][-1].values, freq='d', periods=len(ds_prod['day']))
ds_prod['day'] = time
ds_prod = ds_prod.rename({"day": "time"})
ds = xr.concat([ds_spinup.isel(time=slice(None, -1)), ds_prod], dim="time")

# %% calculate zonal mean precip 
lat_bins = np.linspace(-20, 20, 100)
lat_points = (lat_bins[:-1] + lat_bins[1:]) / 2
ds_zonal = ds.groupby_bins(ds["lat"], lat_bins).mean()

# %% calculate asymmetry 
asym = {}
cummean_asym = {}
for var in ds.data_vars:
    asym[var] = (ds_zonal[var].sel(lat_bins=slice(0, 20)) - np.flip(ds_zonal[var].sel(lat_bins=slice(-20, 0)).values, axis=0)).mean("lat_bins")
    asym[var] = xr.DataArray(
        np.flip(asym[var].values / np.max(np.abs(asym[var].values)), axis=0),
        dims=['DoS'],
        coords={'DoS': np.arange(0, len(asym[var]))},
    )
    cummean_asym[var] = asym[var].cumsum(dim='DoS') / np.arange(1, len(asym[var]) + 1)


# %% plot asymmetry
colors={
    "clivi": "red",
    "pr": "blue"
}
fig, ax = plt.subplots()
ax.axhline(0, color='k', linewidth=0.5)
for var in ds.data_vars:
    ax.plot(asym[var], label=var, color=colors[var], alpha=0.5)
    ax.plot(cummean_asym[var], linestyle='--', color=colors[var], label=f'{var} cummean')

ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel("Day of Simulation")
ax.set_ylabel("Normalised Asymmetric Component")
ax.legend()
fig.savefig('plots/variability/asymmetry_control.png', dpi=300, bbox_inches='tight')

# %% calculate wasserstein distance between north and south for increasing timewindows 
windows = np.arange(10, 130, 10)
distances = []
mask_south = (ds['lat'] > -20) & (ds['lat'] < 0)
mask_north = (ds['lat'] > 0) & (ds['lat'] < 20)

for window in windows:
    print(window)
    # south
    window_south = ds['clivi'].isel(time=slice(-window, None)).where(mask_south).values.flatten()
    window_south = window_south[~np.isnan(window_south)]
    rand_idx = np.random.choice(len(window_south), int(1e7), replace=False)
    window_south = window_south[rand_idx]
    # north
    window_north = ds['clivi'].isel(time=slice(-window, None)).where(mask_north).values.flatten()
    window_north = window_north[~np.isnan(window_north)]
    window_north = window_north[rand_idx]
    # calculate wasserstein distance
    dist = wasserstein_distance(window_south, window_north)
    distances.append(dist)

# %%
fig, ax = plt.subplots()
ax.plot(windows, np.array(distances), marker='o', color='k')
ax.set_xlabel("Window Size / Days")
ax.set_ylabel("Wasserstein Distance / kg m$^{-2}$")
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylim([0, None])
fig.savefig('plots/variability/wasserstein_distance_control_n_s_clivi.png', dpi=300, bbox_inches='tight')

# %%
