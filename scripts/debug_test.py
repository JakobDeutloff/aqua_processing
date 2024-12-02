# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import easygems.healpix as egh
from src.grid_helpers import merge_grid

# %% load data 
ds_2d = xr.open_dataset('/work/bm1183/m301049/icon-mpim/experiments/jed0001/jed0001_atm_2d_19790101T000000Z.14054703_remap.nc').pipe(egh.attach_coords)
ds_3d_main = xr.open_dataset('/work/bm1183/m301049/icon-mpim/experiments/jed0001/jed0001_atm_3d_main_19790101T000000Z.14054703.nc', chunks={}).pipe(merge_grid)
# %% define worldmap function
def worldmap(var, **kwargs):
    projection = ccrs.Robinson()
    fig, ax = plt.subplots(
        figsize=(8, 4), subplot_kw={"projection": projection}, constrained_layout=True
    )
    ax.set_global()
    egh.healpix_show(var, ax=ax, **kwargs)

# %% plot lw_cre and clivi
worldmap(ds_2d['rlut'].isel(time=-1) - ds_2d['rlutws'].isel(time=-1))
worldmap(ds_2d['clivi'].isel(time=-1))

# %% plot lw_cre and sw_cre as a function of clivi
lw_cre = ds_2d['rlutws'].isel(time=-1) - ds_2d['rlut'].isel(time=-1) 
sw_cre = ds_2d['rsutws'].isel(time=-1) - ds_2d['rsut'].isel(time=-1)
bins = np.logspace(-5, 1, 50)
bin_centers = (bins[1:] + bins[:-1]) / 2
sw_cre_binned = sw_cre.groupby_bins(ds_2d['clivi'].isel(time=-1), bins).mean()
lw_cre_binned = lw_cre.groupby_bins(ds_2d['clivi'].isel(time=-1), bins).mean()

fig, ax = plt.subplots()
ax.plot(bin_centers, sw_cre_binned, label='SW', color='blue')
ax.plot(bin_centers, lw_cre_binned, label='LW', color='red')
ax.set_xscale('log')
ax.set_xlabel('IWP / kg m$^{-2}$')
ax.set_ylabel('ICE CRE / W m$^{-2}$')
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
fig.savefig('plots/cres_vs_iwp.png')
# %% plot mean o3 profile in tropics 
mean_o3 = ds_3d_main.o3.where((ds_3d_main.clat < 30) & (ds_3d_main.clat > -30)).isel(time=-1).mean(['ncells'])

# %% 
mean_o3.plot(y='height', yincrease=False)

# %%
ds = xr.open_dataset('/work/bm1183/m301049/icon-mpim/experiments/jed0001/jed0001_atm_2d_19790101T120000Z.14058105.nc')

# %% nh for one day 
n_nodes = 128
compute_time = 0.5
simulation_time = 12/24 

simulated_days_per_day = simulation_time / (compute_time/24)
node_hours_per_day = compute_time * n_nodes / simulation_time
print(f'Node hour per day of simulation: {node_hours_per_day:.2f}')
print(f'Simulated days per day: {simulated_days_per_day:.2f}')

n_nodes_one_month = node_hours_per_day * 31 / 8
print(f'Number of nodes for one month of simulation in 8 h: {n_nodes_one_month:.2f}')

# %%
