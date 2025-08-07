# %%
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

# %% load data
ds = xr.open_dataset(
    "/work/bm1183/m301049/icon_hcap_data/control/production/latlon/atm_2d_latlon.nc"
)
# %%
ds_plot = ds["prw"].isel(time=-1).sel(lon=slice(0, 360), lat=slice(0, 20))
ds_plot = ds_plot.rolling(lat=20, lon=20, center=True).mean()
fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()})
ds_plot.plot.contourf(
    ax=ax,
    transform=ccrs.PlateCarree(),
    levels=np.arange(15, 66, 3),
)
ds_plot.plot.contour(
    ax=ax,
    transform=ccrs.PlateCarree(),
    levels=[36],
    colors="r",
)
# %%
import numpy as np

def haversine(lon1, lat1, lon2, lat2):
    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371.0  # Earth radius in km
    return R * c

# After plotting your contour, extract the vertices as before:
contour = ds_plot.plot.contour(
    ax=ax,
    transform=ccrs.PlateCarree(),
    levels=[38],
    colors="r",
)

length_km = 0
for collection in contour.collections:
    for path in collection.get_paths():
        v = path.vertices  # Nx2 array: [lon, lat]
        lon = v[:, 0]
        lat = v[:, 1]
        # Calculate geodesic distance between consecutive points
        segment_lengths = haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])
        length_km += segment_lengths.sum()

print(f"Contour length (km): {length_km:.2f}")
# %%
