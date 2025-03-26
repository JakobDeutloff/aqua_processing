# %% import necessary libraries
import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar
import os
import sys
import matplotlib.pyplot as plt

# %% load data
run = sys.argv[1]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}

ds = xr.open_dataset(
    f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_20.nc"
).sel(index=slice(0, 1e6))

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
#ds = ds.chunk({"index": 1e6, "height": -1})

# %% determine tropopause height and clearsky
mask_stratosphere = vgrid["zg"].values < 20e3
idx_trop = ds["ta"].where(mask_stratosphere).argmin("height")
height_trop = ds["height"].isel(height=idx_trop)
mask_trop = (ds["height"] > height_trop).load()

# %% build temperature indexer
print("Build temperature indexer")
t_grid = np.linspace(180, 260, 200)

def interpolate_height(ta, height):
    return np.interp(t_grid, ta, height)

# Use Dask to parallelize the interpolation
height_array = xr.apply_ufunc(
    interpolate_height,
    ds["ta"].where(mask_trop),
    ds["height"],
    input_core_dims=[["height"], ["height"]],
    output_core_dims=[["temp"]],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[float],
    dask_gufunc_kwargs={"output_sizes": {"temp": 200}},
)

with ProgressBar(): 
    height_array = height_array.assign_coords(temp=t_grid, index=ds['index']).compute()

# %% regrid to temperature
print("Regrid to temperature")
ds_regrid = ds.interp(height=height_array, method="linear")


# %% save regridded dataset
print("Save dataset")
path = f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_tgrid_20.nc"
if os.path.exists(path):
    os.remove(path)
with ProgressBar():
    ds_regrid.to_netcdf(path)
# %%