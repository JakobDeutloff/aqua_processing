# %%
import xarray as xr
from src.grid_helpers import merge_grid

# %%
ds = xr.open_mfdataset(
    "/work/bm1183/m301049/icon-mpim-4K/experiments/jed0002/jed0002_atm_2d_*.nc",
    chunks={},
).pipe(merge_grid)

# %% calculate tropical mean 
tropical_mean = ds.where((ds.clat < 30) & (ds.clat > -30)).mean('ncells')

# %% save to netcdf
tropical_mean.to_netcdf("/work/bm1183/m301049/icon_hcap_data/plus4K/spinup/jed0002_atm_2d_tropical_mean.nc")
