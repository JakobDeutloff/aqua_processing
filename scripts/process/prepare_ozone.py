# %%
import xarray as xr
from src.grid_helpers import merge_grid, fix_time
import matplotlib.pyplot as plt
import numpy as np
from src.read_data import load_random_datasets, load_definitions

# %%
bc_ozone = xr.open_dataset(
    "/pool/data/ICON/grids/public/mpim/0015/ozone/r0002/bc_ozone_ape.nc"
)
ozone_control = (
    xr.open_dataset(
        "/work/bm1183/m301049/icon-mpim/experiments/jed0111/jed0111_atm_3d_main_19790928T000040Z.16636549.nc"
    )
    .pipe(merge_grid)
    .pipe(fix_time)[["o3", "pfull"]]
).astype(float)
ozone_4K = (
    xr.open_mfdataset(
        "/work/bm1183/m301049/icon-mpim-4K/experiments/jed0222/jed0222_atm_3d_main_*.nc"
    )
    .pipe(merge_grid)
    .pipe(fix_time)[["o3", "pfull"]]
).astype(float)
ozone_2K = (
    xr.open_dataset(
        "/work/bm1183/m301049/icon-mpim-2K/experiments/jed0333/jed0333_atm_3d_main_19791128T000040Z.16501620.nc"
    )
    .pipe(merge_grid)
    .pipe(fix_time)[["o3", "pfull"]]
).astype(float)

datasets = load_random_datasets()
runs, exp_name, colors, labels = load_definitions()

# %% look at daily variability of ozone profile
trop_mean_4K = (
    ozone_4K.isel(time=[0, 30, 59])
    .where((ozone_4K["clat"] > -20) & (ozone_4K["clat"] < 20))
    .mean("ncells")
    .load()
)
# %%
mean_ozone = {}
for run in datasets.keys():
    mean_ozone[run] = datasets[run][["o3", "pfull"]].mean(dim="index")

# %%
fig, ax = plt.subplots(figsize=(10, 5))
for t in trop_mean_4K["time"].values:
    ax.plot(
        trop_mean_4K["o3"].sel(time=t).values.squeeze(),
        trop_mean_4K["pfull"].sel(time=t).values.squeeze(),
        color="k",
        alpha=0.5,
    )
ax.plot(
    mean_ozone["jed0022"]["o3"].values.squeeze(),
    mean_ozone["jed0022"]["pfull"].values.squeeze(),
    color="g",
)
ax.set_yscale("log")

# %% look at mean and std of ozone fields
mask = (bc_ozone["clat"] > np.deg2rad(-20)) & (bc_ozone["clat"] < np.deg2rad(20))
mean_o3 = bc_ozone["O3"].mean(dim="cell")

# %%
mask = (ozone_control["clat"] > -20) & (ozone_control["clat"] < 20)
mean_con_o3 = ozone_control[["o3", "pfull"]].where(mask).mean(dim="ncells")
# std_con_o3 = ozone_control[["o3", "pfull"]].where(mask).std(dim="ncells")

# %%
mask = (ozone_4K["clat"] > -20) & (ozone_4K["clat"] < 20)
mean_4k_o3 = ozone_4K[["o3", "pfull"]].where(mask).mean(dim="ncells")

# %%
mask = (ozone_con_average["clat"] > -20) & (ozone_con_average["clat"] < 20)
mean_con_av_o3 = ozone_con_average.where(mask).mean(dim="ncells")
std_con_av_o3 = ozone_con_average.where(mask).std(dim="ncells")


# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# bc ozone
axes[0].plot(mean_o3.values.squeeze(), mean_o3["plev"], color="k")
axes[0].plot(
    mean_o3.values.squeeze() - std_o3.values.squeeze(),
    mean_o3["plev"],
    color="k",
    linestyle="--",
)
axes[0].plot(
    mean_o3.values.squeeze() + std_o3.values.squeeze(),
    mean_o3["plev"],
    color="k",
    linestyle="--",
)

# control ozone
axes[0].plot(
    mean_con_o3["o3"].values.squeeze(),
    mean_con_o3["pfull"].values.squeeze(),
    color="r",
)
axes[0].plot(
    mean_con_o3["o3"].values.squeeze() - std_con_o3["o3"].values.squeeze(),
    mean_con_o3["pfull"].values.squeeze(),
    color="r",
    linestyle="--",
)
axes[0].plot(
    mean_con_o3["o3"].values.squeeze() + std_con_o3["o3"].values.squeeze(),
    mean_con_o3["pfull"].values.squeeze(),
    color="r",
    linestyle="--",
)

# 4K ozone
axes[0].plot(
    mean_4k_o3["o3"].values.squeeze(),
    mean_4k_o3["pfull"].values.squeeze(),
    color="g",
)

# average ozone
axes[1].plot(
    mean_con_av_o3["o3"].values.squeeze(),
    mean_con_av_o3["pfull"].values.squeeze(),
    color="b",
)
axes[1].plot(
    mean_con_av_o3["o3"].values.squeeze() - std_con_av_o3["o3"].values.squeeze(),
    mean_con_av_o3["pfull"].values.squeeze(),
    color="b",
    linestyle="--",
)
axes[1].plot(
    mean_con_av_o3["o3"].values.squeeze() + std_con_av_o3["o3"].values.squeeze(),
    mean_con_av_o3["pfull"].values.squeeze(),
    color="b",
    linestyle="--",
)

for ax in axes:
    ax.invert_yaxis()
    ax.set_yscale("log")


# %%

# %% bin ozone control data by latitude
lat_bins = np.arange(-90, 90.1, 0.1)
lat_points = (lat_bins[:-1] + lat_bins[1:]) / 2
ozone_control_binned = ozone_control.groupby_bins(
    ozone_control["clat"], lat_bins
).mean()

# %% fill in binned averages into gridded data
lats = ozone_control["clat"].values
bin_indices = np.digitize(lats, lat_bins) - 1

# Get the binned o3 means as a numpy array
binned_o3 = ozone_control_binned["o3"].values.squeeze()
binned_pfull = ozone_control_binned["pfull"].values.squeeze()
o3_gridded = binned_o3[bin_indices, :]
pfull_gridded = binned_pfull[bin_indices, :]
ozone_con_average = xr.Dataset(
    {
        "o3": (["time", "ncells", "height"], o3_gridded[None, :, :]),
        "pfull": (["time", "ncells", "height"], pfull_gridded[None, :, :]),
    },
    coords=ozone_control.coords,
    attrs=ozone_control.attrs,
)
