# %% 
import xarray as xr 
import matplotlib.pyplot as plt
import easygems.healpix as egh
import numpy as np
# %%
ds = xr.open_dataset("/work/bm1183/m301049/icon_hcap_data/control/spinup/jed0001_atm_2d_daymean_hp.nc").pipe(egh.attach_coords)
# %% plot zonal mean precip for every month 
lat_bins = np.linspace(-30, 30, 200)
lat_points = (lat_bins[:-1] + lat_bins[1:]) / 2
precip_zonal = ds['pr'].groupby_bins(ds['lat'], lat_bins).mean()
precip_zonal_monthly = precip_zonal.groupby('time.month').mean(dim='time')

# %% plot zonal mean of precip 
fig, ax = plt.subplots(figsize=(10, 5))
cmap = plt.get_cmap('jet', 6)
for month in range(2, 6):
    ax.plot(lat_points, precip_zonal_monthly.isel(month=month), label=f'Month {month}', color=cmap(month))

ax.set_xlim([-30, 30])
ax.legend()

# %%
