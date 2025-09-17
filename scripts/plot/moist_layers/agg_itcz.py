# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import datetime as dt
from concurrent.futures import ProcessPoolExecutor
from src.grid_helpers import merge_grid, fix_time
from src.moist_layers import plot_prw_field, get_contour_length

# mpl.use("WebAgg")  # Use WebAgg backend for interactive plotting

# %% load data
ds = xr.open_dataset(
    "/work/bm1183/m301049/icon_hcap_data/control/production/latlon/prw_latlon.nc"
)
ds = ds.sel(time=(ds.time.dt.minute == 0) & (ds.time.dt.hour == 0))

# %% check histogram
fig, ax = plt.subplots(figsize=(8, 5))
ds["prw"].isel(time=slice(80, 90)).sel(lat=slice(-40, 40)).plot.hist(
    bins=np.arange(0, 60, 1),
    ax=ax,
    color="grey",
)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig("plots/moist_layers/prw_histogram_40_40.png", dpi=300, bbox_inches="tight")

# %% plot pr vs prw
ds_2d = (
    xr.open_dataset(
        "/work/bm1183/m301049/icon-mpim/experiments/jed0011/jed0011_atm_2d_19790729T000040Z.15371960.nc"
    )
    .pipe(merge_grid)
    .pipe(fix_time)
)

fig, ax = plt.subplots(figsize=(8, 5))
prw_bins = np.arange(0, 71, 1)
binned_pr = (
    (ds_2d["pr"]*60*60*24)
    .isel(time=0)
    .where((ds_2d["clat"] < 20) & (ds_2d["clat"] > -20))
    .groupby_bins(ds_2d["prw"].isel(time=0), bins=prw_bins)
)
binned_pr.mean().plot(ax=ax, color="k", label="Mean PR")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("Water Vapor Path / mm$")
ax.set_ylabel("Precipitation / mm day$^{-1}$")
ax.set_title("")
ax.set_yscale('log')
ax.set_ylim(0, 100)
fig.savefig("plots/moist_layers/pr_vs_prw_20_20.png", dpi=300, bbox_inches="tight")


# %% calculate contourlength 30 mm
def contour_length_for_time(t):
    # Helper for parallel execution
    return int(get_contour_length(ds["prw"].sel(time=t), cont=30))

with ProcessPoolExecutor(max_workers=64) as executor:
    contour_lenghts = list(tqdm(executor.map(contour_length_for_time, ds["time"].values), total=len(ds["time"])))

contour_lenghts_30 = xr.DataArray(
    contour_lenghts / np.max(contour_lenghts),
    dims=["time"],
    coords={"time": ds["prw"].time},
    attrs={"long_name": "Contour length of 30 mm PRW"},
)

# %% calculate contourlength 24 mm
def contour_length_for_time(t):
    # Helper for parallel execution
    return int(get_contour_length(ds["prw"].sel(time=t), cont=24))

with ProcessPoolExecutor(max_workers=64) as executor:
    contour_lenghts = list(tqdm(executor.map(contour_length_for_time, ds["time"].values), total=len(ds["time"])))

contour_lenghts_24 = xr.DataArray(
    contour_lenghts / np.max(contour_lenghts),
    dims=["time"],
    coords={"time": ds["prw"].time},
    attrs={"long_name": "Contour length of 24 mm PRW"},
)

# %% plot contour length 30mm
fig, ax = plt.subplots(figsize=(10, 5))
contour_lenghts_30.plot(ax=ax, color="k", label="Contour length")
ax.set_xlabel("Time")
ax.set_ylabel("Normalized 30 mm contour length")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.axvline(x=dt.datetime.strptime("1979-07-02", "%Y-%m-%d"), color="grey")
ax.axvline(x=dt.datetime.strptime("1979-07-17", "%Y-%m-%d"), color="grey")
fig.savefig("plots/moist_layers/contour_length_30.png", dpi=300, bbox_inches="tight")

# %% plot contour length 24mm
fig, ax = plt.subplots(figsize=(10, 5))
contour_lenghts_24.plot(ax=ax, color="k", label="Contour length")
ax.set_xlabel("Time")
ax.set_ylabel("Normalized 30 mm contour length")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.axvline(x=dt.datetime.strptime("1979-07-14", "%Y-%m-%d"), color="grey")
ax.axvline(x=dt.datetime.strptime("1979-09-23", "%Y-%m-%d"), color="grey")
fig.savefig("plots/moist_layers/contour_length_24.png", dpi=300, bbox_inches="tight")

# %% correlate contour lenths 
corr = xr.corr(contour_lenghts_24, contour_lenghts_30)
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(
    contour_lenghts_24,
    contour_lenghts_30,
    color="k",
    s=5,
)
ax.set_title(f"Correlation: {corr.values:.2f}")
ax.set_xlabel("Normalized 24 mm contour length")
ax.set_ylabel("Normalized 30 mm contour length")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig("plots/moist_layers/contour_length_correlation_trop.png", dpi=300, bbox_inches="tight")

# %%
fig, ax = plot_prw_field(ds["prw"].sel(time="1979-07-02"), 30)
fig.savefig("plots/moist_layers/prw_field_1979-07-02.png", dpi=300, bbox_inches="tight")
fig, ax = plot_prw_field(ds["prw"].sel(time="1979-07-17"), 30)
fig.savefig("plots/moist_layers/prw_field_1979-07-17.png", dpi=300, bbox_inches="tight")


# %%

# %%
