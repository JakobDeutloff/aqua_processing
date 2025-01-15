# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import easygems.healpix as egh

# %% load data 
ds_control = xr.open_dataset('/work/bm1183/m301049/icon_hcap_data/control/spinup/jed0001_atm_2d_daymean_hp.nc', chunks={}).pipe(egh.attach_coords)
ds_4K = xr.open_dataset('/work/bm1183/m301049/icon_hcap_data/plus4K/spinup/jed0002_atm_2d_daymean_hp.nc', chunks={}).pipe(egh.attach_coords)
ds_2K = xr.open_dataset('/work/bm1183/m301049/icon_hcap_data/plus2K/spinup/jed0003_atm_2d_daymean_hp.nc', chunks={}).pipe(egh.attach_coords)

# %% define worldmap function
def worldmap(var, unit=None, **kwargs):
    projection = ccrs.Robinson()
    fig, ax = plt.subplots(
        figsize=(8, 4), subplot_kw={"projection": projection}, constrained_layout=True
    )
    ax.set_global()
    im = egh.healpix_show(var, ax=ax, **kwargs)
    fig.colorbar(im, ax=ax, orientation="horizontal", aspect=40, label=unit)

# %% plot IWP control
worldmap((ds_control['clivi'] + ds_control['qsvi'] + ds_control['qgvi']).sel(time='1979-06-30'), unit='kg m$^{-2}$')

# %% plot IWP 4K 
worldmap((ds_4K['clivi'] + ds_4K['qsvi'] + ds_4K['qgvi']).sel(time='1979-08-31'), unit='kg m$^{-2}$')

# %% plot IWP 2K
worldmap((ds_2K['clivi'] + ds_2K['qsvi'] + ds_2K['qgvi']).sel(time='1979-08-31'), unit='kg m$^{-2}$')
# %% get tropics
ds_trop = ds.isel(time=-1).where((ds.lat < 30) & (ds.lat > -30), drop=True)

# %% calculate lw_cre and sw_cre as a function of clivi
iwp = ds_trop['clivi'] + ds_trop['qsvi'] + ds_trop['qgvi']
lw_cre = ds_trop['rlutws'] - ds_trop['rlut']
sw_cre = ds_trop['rsutws'] - ds_trop['rsut']
bins = np.logspace(-5, 1, 50)
bin_centers = (bins[1:] + bins[:-1]) / 2
sw_cre_binned = sw_cre.groupby_bins(iwp, bins).mean()
lw_cre_binned = lw_cre.groupby_bins(iwp, bins).mean()

# %% plot CRE vs IWP
fig, ax = plt.subplots()
ax.plot(bin_centers, sw_cre_binned, label='SW', color='blue')
ax.plot(bin_centers, lw_cre_binned, label='LW', color='red')
ax.plot(bin_centers, sw_cre_binned + lw_cre_binned, label='NET', color='black')
ax.set_xscale('log')
ax.set_xlabel('IWP / kg m$^{-2}$')
ax.set_ylabel('ICE CRE / W m$^{-2}$')
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
fig.savefig('plots/cres_vs_iwp.png')

# %% plot iwp distribution
fig, ax = plt.subplots()
iwp.plot.hist(bins=np.logspace(-5, 1, 50), ax=ax)
ax.set_xscale('log')
ax.set_xlabel('IWP / kg m$^{-2}$')
ax.set_ylabel('Frequency')

# %%
