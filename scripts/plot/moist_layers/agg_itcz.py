# %%
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from tqdm import tqdm
from src.grid_helpers import merge_grid, fix_time
import datetime as dt

# mpl.use("WebAgg")  # Use WebAgg backend for interactive plotting

# %% load data
ds = xr.open_dataset(
    "/work/bm1183/m301049/icon_hcap_data/control/production/latlon/prw_latlon.nc"
)
ds = ds.sel(time=(ds.time.dt.minute == 0) & (ds.time.dt.hour == 0))

# %% check histogram
fig, ax = plt.subplots(figsize=(8, 5))
ds["prw"].isel(time=slice(80, 90)).sel(lat=slice(-20, 20)).plot.hist(
    bins=np.arange(0, 60, 1),
    ax=ax,
    color="grey",
)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig("plots/moist_layers/prw_histogram_20_20.png", dpi=300, bbox_inches="tight")

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
ax.set_ylim(0, 100)
fig.savefig("plots/moist_layers/pr_vs_prw.png", dpi=300, bbox_inches="tight")

# %% plot precip histogram
fig, ax = plt.subplots(figsize=(8, 5))
ds_2d["pr"].isel(time=slice(80, 90)).where(
    (ds_2d["clat"] < 20) & (ds_2d["clat"] > -20), drop=True
).plot.hist(
    bins=np.arange(0, 0.05, 0.01),
    ax=ax,
    color="grey",
)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("Precipitation / kg m$^{-2}$ s$^{-1}$")
ax.set_ylabel("Frequency")
fig.savefig("plots/moist_layers/pr_histogram_20_20.png", dpi=300, bbox_inches="tight")
# %%
ds_plot = (
    ds["prw"]
    .sel(time="1979-07-11")
    .sel(lon=slice(0, 360), lat=slice(-50, 50))
    .isel(lon=slice(None, None, 10), lat=slice(None, None, 10))
)
pad = 2  # for a window of 5, pad by 2 on each side
ds_plot = ds_plot.pad(lon=(pad, pad), mode="wrap")
ds_plot = ds_plot.rolling(lon=5, lat=5, center=True).mean()

fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()})
ds_plot.plot.contourf(
    ax=ax,
    transform=ccrs.PlateCarree(),
    levels=np.arange(15, 66, 3),
)
ds_plot.plot.contour(
    ax=ax,
    transform=ccrs.PlateCarree(),
    levels=[24],
    colors="r",
)
plt.show()
# %%


def plot_prw_field(ds):
    ds_plot = ds.sel(lon=slice(0, 360), lat=slice(-50, 50)).isel(
        lon=slice(None, None, 10), lat=slice(None, None, 10)
    )
    pad = 2  # for a window of 5, pad by 2 on each side
    ds_plot = ds_plot.pad(lon=(pad, pad), mode="wrap")
    ds_plot = ds_plot.rolling(lon=5, lat=5, center=True).mean()

    fig, ax = plt.subplots(
        figsize=(10, 3), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ds_plot.plot.contourf(
        ax=ax,
        transform=ccrs.PlateCarree(),
        levels=np.arange(15, 66, 3),
    )
    ds_plot.plot.contour(
        ax=ax,
        transform=ccrs.PlateCarree(),
        levels=[24],
        colors="r",
    )
    gl = ax.gridlines(draw_labels=True, linewidth=0)
    gl.top_labels = False
    gl.right_labels = False
    return fig, ax


def haversine(lon1, lat1, lon2, lat2):
    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371.0  # Earth radius in km
    return R * c


def contour(ds):
    pad = 2
    ds_plot = (
        ds.sel(lat=slice(-50, 50))
        .isel(lon=slice(None, None, 10), lat=slice(None, None, 10))
        .pad(lon=(pad, pad), mode="wrap")
        .rolling(lon=5, lat=5, center=True)
        .mean()
    )
    fig, ax = plt.subplots(
        figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    contour = ds_plot.plot.contour(
        ax=ax,
        transform=ccrs.PlateCarree(),
        levels=[24],
        colors="r",
    )
    plt.close(fig)
    # plt.show()
    return contour


def get_contour_length(ds):
    cont = contour(ds)
    total_length = 0.0
    # Iterate over all segments at the first (and only) level
    for segment in cont.allsegs[0]:
        lon = segment[:, 0]
        lat = segment[:, 1]
        segment_lengths = haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])
        total_length += segment_lengths.sum()
    return total_length


# %% calculate contourlength
contour_lenghts = []
for t in tqdm(ds["time"]):
    cont_length = get_contour_length(ds["prw"].sel(time=t))
    contour_lenghts.append(int(cont_length))

contour_lenghts = xr.DataArray(
    contour_lenghts / np.max(contour_lenghts),
    dims=["time"],
    coords={"time": ds["prw"].time},
    attrs={"long_name": "Contour length of 24 mm PRW"},
)

# %% plot contour length
fig, ax = plt.subplots(figsize=(10, 5))
contour_lenghts.plot(ax=ax, color="k", label="Contour length")
ax.set_xlabel("Time")
ax.set_ylabel("Normalized 24 mm contour length")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.axvline(x=dt.datetime.strptime("1979-07-14", "%Y-%m-%d"), color="grey")
ax.axvline(x=dt.datetime.strptime("1979-09-23", "%Y-%m-%d"), color="grey")
fig.savefig("plots/moist_layers/contour_length.png", dpi=300, bbox_inches="tight")

# %%
fig, ax = plot_prw_field(ds["prw"].sel(time="1979-07-14"))
fig.savefig("plots/moist_layers/prw_field_1979-07-14.png", dpi=300, bbox_inches="tight")
fig, ax = plot_prw_field(ds["prw"].sel(time="1979-09-23"))
fig.savefig("plots/moist_layers/prw_field_1979-09-23.png", dpi=300, bbox_inches="tight")


# %%
