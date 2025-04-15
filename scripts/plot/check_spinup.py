# %%
import xarray as xr
import matplotlib.pyplot as plt
import easygems.healpix as egh
import numpy as np
import matplotlib as mpl
import pandas as pd
from scipy.stats import wasserstein_distance

# %%
ds = xr.open_dataset(
    f"/work/bm1183/m301049/icon_hcap_data/plus2K/spinup/jed0003_atm_2d_daymean_hp.nc"
).pipe(egh.attach_coords)['pr']

# %% calculate wasserstein distances statistics
mask = (ds['lat'] > -20) & (ds['lat'] < 20)
window = 10
ref_window = ds.isel(time=slice(-window, None)).where(mask).values.flatten()
ref_window = ref_window[~np.isnan(ref_window)]
rand_idx = np.random.choice(len(ref_window), int(1e6), replace=False)
ref_window = ref_window[rand_idx]
dists = []
for i in range(int(len(ds.time)/window)):
    print(i)
    start = int(i * window)
    end = int((i + 1) * window)
    test_window = ds.isel(time=slice(start, end)).where(mask).values.flatten()
    test_window = test_window[~np.isnan(test_window)]
    test_window = test_window[rand_idx]
    dist = wasserstein_distance(test_window, ref_window)
    dists.append(dist)

dists = xr.DataArray(
    dists,
    dims=['time'],
    coords={'time': ds.time[5::window].values},
)

# %% calculate mean and std 
mean = ds.where(mask).mean('cell')
std = ds.where(mask).std('cell')
# %% plot metrics 
fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
axes[0].plot(mean.time, mean*60*60*24, label='mean', color='k')
axes[0].set_ylabel("Mean Precipitation / mm day$^{-1}$")
axes[1].plot(std.time, std*60*60*24, label='std', color='k')
axes[1].set_ylabel("Std Precipitation / mm day$^{-1}$")
axes[2].plot(dists.time, dists*60*60*24, label='wasserstein distance', color='k')
axes[2].set_ylabel("Wasserstein Distance / mm day$^{-1}$")
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
axes[-1].set_ylim([0, None])
fig.tight_layout()
axes[-1].tick_params(axis='x', rotation=45)
fig.savefig('plots/variability/wasserstein_distance_2K_10d.png', dpi=300, bbox_inches='tight')

# %%
