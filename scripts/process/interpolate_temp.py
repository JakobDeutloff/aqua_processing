# %% import necessary libraries
import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar
import os
import sys
import matplotlib.pyplot as plt
from src.calc_variables import calc_stability_t, calc_pot_temp, calc_heating_rates_t, calc_w_sub_t, calc_conv_t

# %% load data
run = sys.argv[1]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K", "jed2224": "const_o3"}

ds = xr.open_dataset(
    f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed_64.nc"
).sel(index=slice(0, 5e6))

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
ds = ds.chunk({"index": 5e3, "height": -1})
ds = ds.astype(float)

# %% determine tropopause height and clearsky
mask_stratosphere = vgrid["zg"].values < 20e3
idx_trop = ds["ta"].where(mask_stratosphere).argmin("height")
height_trop = ds["height"].isel(height=idx_trop)
mask_trop = (ds["height"] > height_trop).load()

# %% build temperature indexer
print("Build temperature indexer")
t_grid = np.linspace(180, 260, 60)

def interpolate_height(ta, height):
    return np.interp(t_grid, ta[~np.isnan(ta)], height[~np.isnan(ta)], left=np.nan, right=np.nan)

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
    dask_gufunc_kwargs={"output_sizes": {"temp": 60}},
)

with ProgressBar(): 
    height_array = height_array.assign_coords(temp=t_grid, index=ds['index']).compute()

# %% regrid to temperature
print("Regrid to temperature")
with ProgressBar():
    ds_regrid = ds.interp(height=height_array, method="quintic").compute()

# %% calculate stability
theta = calc_pot_temp(ds_regrid["ta"], ds_regrid["pfull"])
stab = calc_stability_t(
    theta,
    ds_regrid["ta"],
    ds_regrid["zg"],
)
hr = calc_heating_rates_t(
    ds_regrid["rho"],
    ds_regrid["rsd"] - ds_regrid["rsu"],
    ds_regrid["rld"] - ds_regrid["rlu"],
    ds_regrid["zg"],
)
sub = calc_w_sub_t(hr["net_hr"], stab)
conv = calc_conv_t(sub)


# %% control plot of temp and stability 
fig, axes = plt.subplots(1, 4, figsize=(10, 8), sharey=True)

axes[0].plot(
    hr['net_hr'].where(hr['temp']>200).mean('index'),
    hr['temp']
)
axes[0].set_xlabel("Net heating rate / K/day")
axes[1].plot(
    stab.where(stab['temp']>200).mean('index'),
    stab['temp']
)
axes[1].set_xlabel("Stability / K/K")
axes[2].plot(
    sub.where(sub['temp']>200).median('index'),
    sub['temp']
)
axes[2].set_xlabel("Subsidence / K/day")
axes[3].plot(
    conv.where(conv['temp']>200).median('index'),
    conv['temp']
)
axes[3].set_xlabel("Convergence / 1/day")

axes[0].invert_yaxis()
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim([260, 200])

fig.savefig(f'plots/misc/t_tinterp_{run}.png', dpi=300, bbox_inches='tight')

# %% save regridded dataset
print("Save dataset")
path = f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_tgrid_5e6.nc"
if os.path.exists(path):
    os.remove(path)
with ProgressBar():
    ds_regrid.to_netcdf(path)
# %%