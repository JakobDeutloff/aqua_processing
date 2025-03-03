# %%
import xarray as xr
from src.grid_helpers import merge_grid
import numpy as np
import matplotlib.pyplot as plt

# %%
pr_control = xr.open_mfdataset(
    "/work/bm1183/m301049/icon-mpim/experiments/jed0011/jed0011_atm_2d_daymean*.nc",
).pipe(merge_grid)['pr']
pr_2K = xr.open_mfdataset(
    "/work/bm1183/m301049/icon-mpim-2K/experiments/jed0033/jed0033_atm_2d_daymean*.nc",
).pipe(merge_grid)['pr']
pr_4K = xr.open_mfdataset(
    "/work/bm1183/m301049/icon-mpim-4K/experiments/jed0022/jed0022_atm_2d_daymean*.nc",
).pipe(merge_grid)['pr']

# %% bin by latitude
lat_bins = np.arange(-90, 90.5, 0.5)
lat_points = (lat_bins[:-1] + lat_bins[1:]) / 2
pr_control_binned = pr_control.groupby_bins("clat", lat_bins).mean().mean('time').load()
pr_2K_binned = pr_2K.groupby_bins("clat", lat_bins).mean().mean('time').load()
pr_4K_binned = pr_4K.groupby_bins("clat", lat_bins).mean().mean('time').load()

# %% plot binned precip
fig, ax = plt.subplots()

ax.plot(lat_points, pr_control_binned*60*60*24, label="control", color='k')
ax.plot(lat_points, pr_2K_binned*60*60*24, label="2K", color='orange')
ax.plot(lat_points, pr_4K_binned*60*60*24, label="4K", color='red')

ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel("Latitude")
ax.set_ylabel("Precipitation / mm/day")
ax.legend()
ax.set_xlim([-30, 30])
fig.savefig('plots/precip.png', dpi=300, bbox_inches='tight')


# %%
