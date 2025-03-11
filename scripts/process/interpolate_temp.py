# %% import necessary libraries
import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar
import os
import sys
from tqdm import tqdm

# %% load data
run = "jed0011"  # sys.argv[1]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}

ds = xr.open_dataset(
    f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample.nc"
).sel(index=slice(0, 1e3))

vgrid = (
    xr.open_dataset(
        "/work/bm1183/m301049/icon-mpim/experiments/jed0001/atm_vgrid_angel.nc"
    )
    .mean("ncells")
    .rename({"height": "height_2", "height_2": "height"})
)


# drop all variables that do not contain height as a dimension
ds = ds.drop_vars([var for var in ds.variables if "height" not in ds[var].dims])
ds = ds.assign(zg = vgrid["zg"])
ds = ds.assign(dzghalf = vgrid["dzghalf"])
ds = ds.assign_coords(index = ds["index"])



# %% determine tropopause height and clearsky
mask_stratosphere = vgrid["zg"].values < 20e3
idx_trop = ds["ta"].where(mask_stratosphere).argmin("height")
height_trop = ds["height"].isel(height=idx_trop)
mask_trop = ds["height"] > height_trop

# %% build temperature indexer
print("Build temperature indexer")
t_grid = np.linspace(200, 260, 121)

height_array = xr.DataArray(
    np.zeros((len(ds["index"]), len(t_grid))) * np.nan,
    dims=["index", "temp"],
    coords={"index":ds['index'], "temp": t_grid},
)

for i in tqdm(ds["index"]):
    height_array.loc[i] = np.interp(
        t_grid, ds["ta"].sel(index=i).where(mask_trop.sel(index=1)), ds["height"]
    )

# %% regrid to temperature
print("Regrid to temperature")
ds_regrid = ds.interp(height=height_array, method="quintic")

# %% save regridded dataset
print("Save dataset")
path = f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_tgrid.nc"
if os.path.exists(path):
    os.remove(path)
with ProgressBar():
    ds_regrid.to_netcdf(path)
# %%
