# %%
import healpix as hp
import numpy as np
import xarray as xr
import easygems.remap as egr
from src.grid_helpers import merge_grid

# %% load dataset
ds = xr.open_dataset(
    "/work/bm1183/m301049/icon-mpim/experiments/jed0001/jed0001_atm_2d_19790101T000000Z.14054703.nc"
).pipe(merge_grid)

#%% convert clat and clon from radian to degree 
ds = ds.assign(clat = np.degrees(ds.clat), clon = np.degrees(ds.clon))

# %% create healpix grid
order = zoom = 10
nside = hp.order2nside(order)
npix = hp.nside2npix(nside)

hp_lon, hp_lat = hp.pix2ang(nside=nside, ipix=np.arange(npix), lonlat=True, nest=True)
hp_lon = (hp_lon + 180) % 360 - 180  # [-180, 180)
hp_lon += 360 / (4 * nside) / 4  # shift quarter-width

# %% load interpolation weigths
weights = egr.compute_weights_delaunay((ds.clon, ds.clat), (hp_lon, hp_lat))

# %% save weights
weights.to_netcdf("weights_z10.nc")