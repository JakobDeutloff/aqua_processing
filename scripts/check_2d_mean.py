# %% import 
import xarray as xr 
import matplotlib.pyplot as plt

#%% load data
ds = xr.open_dataset('/work/bm1183/m301049/icon-mpim/experiments/jed0001/jed0001_atm_2d_daymean_19790131T000000Z.14218015_tropmean.nc')
ds_full = xr.open_dataset('/work/bm1183/m301049/icon-mpim/experiments/jed0001/jed0001_atm_2d_daymean_19790131T000000Z.14218015.nc')
# %% plot timeseries of main variables
main_vars = ['tas', 'ps', 'pr']
fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharex=True)
for i, var in enumerate(main_vars):
    ax = axes[i]
    ds[var].plot(ax=ax)
    ax.set_title(var)
    ax.set_ylabel(ds_full[var].units)
fig.tight_layout()

# %% plot timeseries of radiative fluxes at toa
rad_vars_t = ['rsdt', 'rsut', 'rlut', 'rsutcs', 'rlutcs', 'rsutws', 'rlutws']
fig, axes = plt.subplots(2, 4, figsize=(10, 7), sharex=True)
for i, var in enumerate(rad_vars_t):
    ax = axes.flat[i]
    ds[var].plot(ax=ax)
    ax.set_title(var)
    ax.set_ylabel(ds_full[var].units)
fig.tight_layout()

# %% plot timeseries of radiative fluxes at surface
rad_vars_d = ['rsds', 'rsus', 'rlds', 'rlus', 'rsdscs', 'rsuscs', 'rldscs']
fig, axes = plt.subplots(2, 4, figsize=(10, 7), sharex=True)
for i, var in enumerate(rad_vars_d):
    ax = axes.flat[i]
    ds[var].plot(ax=ax)
    ax.set_title(var)
    ax.set_ylabel(ds_full[var].units)
fig.tight_layout()

# %% plot timeseries of cloud variables
cloud_vars = ['qrvi', 'cllvi', 'clivi', 'qsvi', 'qgvi', 'clt', 'prw']
fig, axes = plt.subplots(2, 4, figsize=(10, 7), sharex=True)
for i, var in enumerate(cloud_vars):
    ax = axes.flat[i]
    ds[var].plot(ax=ax)
    ax.set_title(var)  
    ax.set_ylabel(ds_full[var].units)
fig.tight_layout()

# %% plot timeseries of surface winds and heat fluxes
wind_vars = ['uas', 'vas', 'sfcwind', 'hfss', 'hfls', 'evspsbl', 'hus2m']
fig, axes = plt.subplots(2, 4, figsize=(10, 7), sharex=True)
for i, var in enumerate(wind_vars):
    ax = axes.flat[i]
    ds[var].plot(ax=ax)
    ax.set_title(var)
    ax.set_ylabel(ds_full[var].units)
fig.tight_layout()


# %%
