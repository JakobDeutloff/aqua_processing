# %%
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from tqdm import tqdm

# %% load data
ds = xr.open_dataset(
    "/work/bm1183/m301049/icon_hcap_data/control/production/latlon/prw_latlon.nc"
)
ds = ds.sel(time=(ds.time.dt.minute == 0) & (ds.time.dt.hour == 0))

# %% check histogram
ds["prw"].isel(time=slice(100, 110)).sel(lat=slice(-40, 40)).plot.hist(
    bins=np.arange(0, 60, 1)
)
# %%
ds_plot = (
    ds["prw"]
    .isel(time=10)
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
# %%

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
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()})
    contour = ds_plot.plot.contour(
        ax=ax,
        transform=ccrs.PlateCarree(),
        levels=[24],
        colors="r",
    )
    plt.close(fig)
    return contour

def get_contour_length(ds):
    cont = contour(ds)
    v = cont.allsegs[0][0] 
    lon = v[:, 0]
    lat = v[:, 1]
    # Calculate geodesic distance between consecutive points
    segment_lengths = haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])
    if np.min(segment_lengths) < 0:
        raise ValueError("Negative segment length found, check data.")
    length_km = segment_lengths.sum()
    return length_km


# %% calculate cntourlength
contour_lenghts_2 = []
for t in tqdm(ds["time"]):
    cont_length = get_contour_length(ds['prw'].sel(time=t))
    contour_lenghts_2.append(int(cont_length))

contour_lenghts_2 = xr.DataArray(
    contour_lenghts_2/np.max(contour_lenghts_2),
    dims=["time"],
    coords={"time": ds["prw"].time},
    attrs={
        "long_name": "Contour length of 24 mm PRW"}
)

# %% plot contour length
fig, ax = plt.subplots(figsize=(10, 5))
contour_lenghts_2.plot(ax=ax, color="r", label="Contour length")
ax.set_xlabel("Time")
ax.set_ylabel("Normalized contour length")
ax.legend()

# %%
