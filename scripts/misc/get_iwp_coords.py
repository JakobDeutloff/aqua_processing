# %%
import xarray as xr
from src.grid_helpers import merge_grid
import numpy as np
from src.sampling import sample_profiles

# %% naming
runs = ["jed0011"]
model_config = {
    "jed0011": "icon-mpim",
    "jed0022": "icon-mpim-4K",
    "jed0033": "icon-mpim-2K",
}
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}

# %% import 2D data and subsample
run = "jed0011"
ds_3D = xr.open_mfdataset(
    f"/work/bm1183/m301049/{model_config[run]}/experiments/{run}/{run}_atm_3d_main_19*.nc",
    chunks={},
).pipe(merge_grid)

ds_2D = xr.open_mfdataset(
    f"/work/bm1183/m301049/{model_config[run]}/experiments/{run}/{run}_atm_2d_19*.nc",
    chunks={},
).pipe(merge_grid)

ds_2D = ds_2D.sel(time=ds_3D.time)

# %% select tropics 
ds_2D_trop = ds_2D.where((ds_2D.clat < 30) & (ds_2D.clat > -30), drop=True)

# %% calculate IWP 
ds_2D_trop = ds_2D_trop.assign(IWP = ds_2D_trop['clivi'] + ds_2D_trop['qsvi'] + ds_2D_trop['qgvi'])

# %% calculate local time from longitude
ds_2D_trop = ds_2D_trop.assign(time_local=lambda d: d.time.dt.hour + d.clon / 15)
ds_2D_trop["time_local"] = ds_2D_trop["time_local"].where(ds_2D_trop["time_local"] < 24, ds_2D_trop["time_local"] - 24)
ds_2D_trop["time_local"].attrs = {"units": "h", "long_name": "Local time"}

# %% load data to memory
ds_2D_trop = ds_2D_trop[["IWP", "time_local"]].load()

# %% define bins 
iwp_bins = np.logspace(-6, np.log10(40), 101)
iwp_points = (iwp_bins[:-1] + iwp_bins[1:]) / 2
time_bins = np.linspace(0, 24, 25)
time_points = (time_bins[:-1] + time_bins[1:]) / 2
n_samples = 50

# %% initialize coordinates dataset 
iwp_coords = xr.Dataset(
    {
        "time": xr.DataArray(data=np.zeros([len(iwp_points), len(time_points), n_samples]), dims=["ciwp", "ctime", "index"]),
        "ncells": xr.DataArray(data=np.zeros([len(iwp_points), len(time_points), n_samples]), dims=["ciwp", "ctime", "index"]),

    },
    coords={"ciwp": iwp_points, "ctime": time_points, "index": np.arange(n_samples)},
)
        
# %%
iwp_coords = sample_profiles(ds_2D_trop, time_bins, time_points, iwp_bins, iwp_points, n_samples, iwp_coords)

# %% save to file
iwp_coords.to_netcdf(
    f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/iwp_sample/iwp_coords.nc"
)
