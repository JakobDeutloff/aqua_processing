# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# %% load data
ds = xr.open_dataset(
    f"/work/bm1183/m301049/icon_hcap_data/plus2K/production/random_sample/jed0033_randsample_processed_20_conn.nc"
).sel(index=slice(None, 1e6))
vgrid = xr.open_dataset(
    "/work/bm1183/m301049/icon-mpim/experiments/jed0001/atm_vgrid_angel.nc"
).mean("ncells").rename({"height_2": "height", "height": "height_2"})

# %% calculate brightness temp from fluxes
def calc_brightness_temp(flux):
    return (flux / 5.67e-8) ** (1 / 4)

# %% determine tropopause height and clearsky
mask_stratosphere = vgrid["zg"].values < 25e3
idx_trop = ds["ta"].where(mask_stratosphere).argmin("height")
height_trop = ds["height"].isel(height=idx_trop)
mask_trop = (ds["height"] > height_trop)

#%% only look at clouds with cloud tops above 350 hPa and IWP > 1e-1 so that e = 1 can be assumed
mask = (ds["iwp"] > 1e-1) 
T_bright = calc_brightness_temp(ds['rlut'])
height_bright_idx = np.abs(ds['ta'].where(mask_trop) - T_bright).argmin("height")  # fills with 0 for NaNs
ice_cumsum = ds["iwc_cumsum"].isel(height=height_bright_idx)
mean_ice_cumsum = ice_cumsum.where(mask).median()
print(mean_ice_cumsum.values)

# %%
ice_cumsum.where(mask).plot.hist(bins=np.linspace(0, 0.1, 20))


# %% plot profiles of all clouds with iwc_cumsum==0
mask_zero = mask & (ice_cumsum == 0)
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

axes[0].plot(
    (ds['cli'] + ds['qs'] + ds['qg']).where(mask_zero).mean("index"),
    vgrid['zg']/1e3,
)
axes[0].axhline(
    vgrid['zg'].isel(height=int(height_bright_idx.where(mask_zero).mean().values)).values/1e3,
)
axes[1].plot(
  (ds['cli'] + ds['qs'] + ds['qg']).where(~mask_zero).mean("index"),
    vgrid['zg']/1e3,
)
axes[1].axhline(
    vgrid['zg'].isel(height=int(height_bright_idx.where(~mask_zero).mean().values)).values/1e3,
)


# %% compare fluxes 
h_emm = np.abs(ds['iwc_cumsum'] - mean_ice_cumsum.values).argmin('height')
T_hc = ds['ta'].isel(height=h_emm)
flux = T_hc**4 * 5.67e-8

fig, ax = plt.subplots()
(flux-ds['rlut']).where(mask).groupby_bins(ds['iwp'],  bins=np.logspace(-1, np.log10(40), 51)).mean().plot(ax=ax)
ax.set_xscale('log')

# %%
