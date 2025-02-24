# %% 
import xarray as xr
from src.grid_helpers import merge_grid
import numpy as np
import matplotlib.pyplot as plt

# %% 
ds = xr.open_dataset('/work/bm1183/m301049/icon-mpim/experiments/jed0011/jed0011_atm_2d_19790709T000040Z.15356915.nc', chunks={'time':-1, 'ncells':-1}).pipe(merge_grid)

# %% bin by latitude 
lat_bins = np.arange(-90, 90.5, 0.5)
lat_points = (lat_bins[:-1] + lat_bins[1:]) / 2
pr_binned = ds['pr'].groupby_bins(ds['clat'], lat_bins).mean()

# %% plot binned precip 
fig, ax = plt.subplots()
for time in pr_binned.time:
    ax.plot(lat_points, pr_binned.sel(time=time), label=time.values, color='grey', alpha=0.5)

ax.plot(lat_points, pr_binned.mean('time'), label='mean', color='black')

ax.set_xlim([-30, 30])
# %%
