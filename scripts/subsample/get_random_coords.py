# %%
import xarray as xr
import numpy as np
import dask.array as da
from src.grid_helpers import merge_grid

# %% load 2D data
runs = ["jed0011", "jed0022", "jed0033"]
model_config = {
    "jed0011": "icon-mpim",
    "jed0022": "icon-mpim-4K",
    "jed0033": "icon-mpim-2K",
}
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}

# %%
for run in runs:
    path = f"/work/bm1183/m301049/{model_config[run]}/experiments/{run}/"
    ds_2D = xr.open_mfdataset(f"{path}{run}_atm_3d_main_19*.nc", chunks={}).pipe(
        merge_grid
    )

    # select tropics
    ds_2D_trop = ds_2D.where((ds_2D.clat < 30) & (ds_2D.clat > -30), drop=True)

    # get random coordinates across time and ncells
    ncells = ds_2D_trop.sizes["ncells"]
    time = ds_2D_trop.sizes["time"]

    # Generate unique pairs of random indices
    num_samples = int(1e5)
    total_indices = ncells * time
    random_indices = da.random.randint(0, total_indices, num_samples).compute()

    random_ncells_idx = random_indices % ncells
    random_time_idx = random_indices // ncells

    # create xrrays
    random_coords = xr.Dataset(
        {
            "time": xr.DataArray(ds_2D_trop.time[random_time_idx].values, dims="index"),
            "ncells": xr.DataArray(
                ds_2D_trop.ncells[random_ncells_idx].values, dims="index"
            ),
        },
        coords={"index": np.arange(num_samples)},
    )

    #  save to file
    random_coords.to_netcdf(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/random_coords.nc"
    )

# %%
