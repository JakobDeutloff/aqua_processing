# %% import 
import xarray as xr 
import matplotlib.pyplot as plt

#%% load data
ds_2k = xr.open_dataset('/work/bm1183/m301049/icon_hcap_data/plus2K/spinup/jed0003_atm_2d_tropical_mean.nc')
ds_4k = xr.open_dataset('/work/bm1183/m301049/icon_hcap_data/plus4K/spinup/jed0002_atm_2d_tropical_mean.nc')
ds_control = xr.open_dataset('/work/bm1183/m301049/icon_hcap_data/control/spinup/jed0001_atm_2d_daymean_tropical_mean.nc')
ds_full = xr.open_dataset('/work/bm1183/m301049/icon-mpim-4K/experiments/jed0002/jed0002_atm_2d_daymean_19790715T000000Z.14423832.nc')
# %% plot timeseries of all variables
def plot_var(varname, ax):
    ds_control[varname].isel(time=slice(1, None)).plot(ax=ax, label='control', color='black')
    ds_2k[varname].plot(ax=ax, label='plus2K', color='orange')
    ds_4k[varname].plot(ax=ax, label='plus4K', color='red')
    ax.set_title(ds_full[varname].attrs['long_name'])
    ax.set_ylabel(ds_full[varname].attrs['units'])

fig, axes = plt.subplots(8, 4, figsize=(30, 30), sharex='col')

for i, varname in enumerate(ds_full.data_vars):
    plot_var(varname, axes.flat[i])

fig.savefig('plots/spinup_tropical_mean_timeseries.png', dpi=300, bbox_inches='tight')

# %%
