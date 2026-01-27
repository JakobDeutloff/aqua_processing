# %%
import xarray as xr
from src.grid_helpers import fix_time

# %% load raw icon data
ds_vel = xr.open_dataset(
    "/work/bu1562/m301049/icon-mpim/experiments/jed0111/jed0111_atm_3d_vel_19790919T000040Z.16632651.nc"
).sel(height_2=64)[["wa"]]
ds_main = xr.open_dataset(
    "/work/bu1562/m301049/icon-mpim/experiments/jed0111/jed0111_atm_3d_main_19790919T000040Z.16632651.nc"
).sel(height=64)[["rho"]]
ds_icon = xr.merge([ds_vel, ds_main])
ds_icon.to_netcdf(
    "/work/bm1183/m301049/icon_hcap_data/publication/vertical_vel/vel_icon_2.nc"
)

# %%
