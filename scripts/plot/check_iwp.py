# %%
import xarray as xr
import numpy as np
from src.grid_helpers import merge_grid
import matplotlib.pyplot as plt

# %% load data
path = "/work/bm1183/m301049/"
ds_control = xr.open_mfdataset(
    path + "icon-mpim/experiments/jed0011/jed0011_atm_2d_19790730T000000Z.14380341.nc",
    chunks={},
).pipe(merge_grid)
ds_4K = xr.open_mfdataset(
    path
    + "icon-mpim-4K/experiments/jed0022/jed0022_atm_2d_19790930T000000Z.14795929.nc",
    chunks={},
).pipe(merge_grid)
ds_2K = xr.open_mfdataset(
    path
    + "icon-mpim-2K/experiments/jed0033/jed0033_atm_2d_19790930T000000Z.15016798.nc",
    chunks={},
).pipe(merge_grid)



# %%  calculate iwp in tropics
def get_tropics(ds):
    return ds.where((ds.clat > -30) & (ds.clat < 30), drop=True)


iwp_control = (
    get_tropics(ds_control["qsvi"])
    + get_tropics(ds_control["qgvi"])
    + get_tropics(ds_control["clivi"])
).load()
iwp_2K = (
    get_tropics(ds_2K["qsvi"])
    + get_tropics(ds_2K["qgvi"])
    + get_tropics(ds_2K["clivi"])
).load()
iwp_4K = (
    get_tropics(ds_4K["qsvi"])
    + get_tropics(ds_4K["qgvi"])
    + get_tropics(ds_4K["clivi"])
).load()

# %% calculate iwp hist
bins = np.logspace(-6, 1, 70)
hist_control, _ = np.histogram(iwp_control, bins=bins, density=False)
hist_control = hist_control / iwp_control["ncells"].size
hist_2K, _ = np.histogram(iwp_2K, bins=bins, density=False)
hist_2K = hist_2K / iwp_2K["ncells"].size
hist_4K, edges = np.histogram(iwp_4K, bins=bins, density=False)
hist_4K = hist_4K / iwp_4K["ncells"].size

# %% plot iwp hists
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.stairs(hist_control, edges, label="control", color='k')
ax.stairs(hist_2K, edges, label="2K", color='orange')
ax.stairs(hist_4K, edges, label="4K", color='r')

ax.legend()
ax.set_xscale("log")
ax.set_ylabel('P(IWP)')
ax.set_xlabel('IWP / kg m$^{-2}$')
ax.spines[['top', 'right']].set_visible(False)
fig.savefig('iwp_hist.png', dpi=300)
# %%
ds_trop = get_tropics(xr.concat([ds_2K, ds_2K_last], 'time'))

# %%
ds_mean = ds_trop.mean('ncells')

# %% plot time series
def plot_var(varname, ax):
    ds_mean[varname].plot(ax=ax)


fig, axes = plt.subplots(8, 4, figsize=(30, 30), sharex='col')

for i, varname in enumerate(ds_mean.data_vars):
    plot_var(varname, axes.flat[i])

# %%
