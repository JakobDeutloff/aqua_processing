# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# %%
bc_ozone = xr.open_dataset(
    "/pool/data/ICON/grids/public/mpim/0015/ozone/r0002/bc_ozone_ape.nc"
)
ozone_control = xr.open_dataset(
    "/work/bm1183/m301049/icon_hcap_data/control/production/ozone_control.nc"
)

# %% bin ozone control data by latitude
lat_bins = np.arange(-90, 90.1, 0.1)
lat_points = (lat_bins[:-1] + lat_bins[1:]) / 2
ozone_binned = ozone_control.groupby_bins(ozone_control["clat"], lat_bins).mean()

# %% interpolate to pressure
p_grid = bc_ozone["plev"].values


def interpolate_pressure(p, height):
    return np.interp(p_grid, p[~np.isnan(p)], height[~np.isnan(p)])


# Use Dask to parallelize the interpolation
height_array = xr.apply_ufunc(
    interpolate_pressure,
    ozone_binned["pfull"],
    ozone_binned["height"],
    input_core_dims=[["height"], ["height"]],
    output_core_dims=[["pfull"]],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[float],
    dask_gufunc_kwargs={"output_sizes": {"pfull": 59}},
)

height_array = height_array.assign_coords(
    pfull=p_grid, clat_bins=ozone_binned["clat_bins"]
).compute()
ozone_binned_regrid = ozone_binned.interp(
    height=height_array, method="linear"
).compute()

# %% fill in binned averages into gridded data
lats = ozone_control["clat"].values
bin_indices = np.digitize(lats, lat_bins) - 1

# Get the binned o3 means as a numpy array
binned_o3 = ozone_binned_regrid["o3"].values.squeeze()
o3_gridded = binned_o3[bin_indices, :]
ozone_average = bc_ozone.copy(deep=True)
ozone_average["O3"] = (
    ("time", "cell", "plev"),
    o3_gridded[None, :, :].astype("float64"),
)
ozone_average["O3"] = ozone_average["O3"].transpose("time", "plev", "cell")
# %% compare tropical mean ozone from control to bc
tropical_mean_control = (
    ozone_average["O3"]
    .where(
        (ozone_average["clat"] > np.deg2rad(-20))
        & (ozone_average["clat"] < np.deg2rad(20))
    )
    .mean("cell")
)
tropical_std_control = (
    ozone_average["O3"]
    .where(
        (ozone_average["clat"] > np.deg2rad(-20))
        & (ozone_average["clat"] < np.deg2rad(20))
    )
    .std("cell")
)
tropical_mean_bc = (
    bc_ozone["O3"]
    .where((bc_ozone["clat"] > np.deg2rad(-20)) & (bc_ozone["clat"] < np.deg2rad(20)))
    .mean("cell")
)
tropical_std_bc = (
    bc_ozone["O3"]
    .where((bc_ozone["clat"] > np.deg2rad(-20)) & (bc_ozone["clat"] < np.deg2rad(20)))
    .std("cell")
)

# %%
clat_bin_mids = np.array(
    [interval.mid for interval in ozone_binned["clat_bins"].values]
)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(
    tropical_mean_control.values.squeeze(),
    tropical_mean_control["plev"] / 100,
    label="Control",
    color="blue",
)
ax.plot(
    tropical_mean_control.values.squeeze() + tropical_std_control.values.squeeze(),
    tropical_mean_control["plev"] / 100,
    linestyle="--",
    color="blue",
)

ax.plot(
    tropical_mean_bc.values.squeeze(),
    tropical_mean_bc["plev"] / 100,
    label="BC Ozone",
    color="k",
)
ax.plot(
    tropical_mean_bc.values.squeeze() + tropical_std_bc.values.squeeze(),
    tropical_mean_bc["plev"] / 100,
    linestyle="--",
    color="k",
)
ax.invert_yaxis()
ax.set_yscale("log")

# %% save ozone in experiment folder
ozone_average["O3"] = ozone_average["O3"].astype("float32")
ozone_average.to_netcdf(
    "/work/bm1183/m301049/icon-mpim-4K/experiments/jed0224/control_ozone.nc"
)

# %%
