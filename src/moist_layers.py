# %%
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


# %%

def plot_prw_field(ds, cont=24):
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
        levels=[cont],
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


def contour(ds, cont):
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
        levels=[cont],
        colors="r",
    )
    plt.close(fig)
    return contour


def get_contour_length(ds, cont=24):
    cont = contour(ds, cont)
    total_length = 0.0
    # Iterate over all segments at the first (and only) level
    for segment in cont.allsegs[0]:
        lon = segment[:, 0]
        lat = segment[:, 1]
        segment_lengths = haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])
        total_length += segment_lengths.sum()
    return total_length
