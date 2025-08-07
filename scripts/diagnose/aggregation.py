# %%
import cloudmetrics as cm
import xarray as xr
import matplotlib.pyplot as plt
from scipy import ndimage
from src.grid_helpers import fix_time
from src.read_data import load_definitions
import numpy as np
import cartopy.crs as ccrs

# %% load data
runs, exp_name, colors, line_labels, sw_color, lw_color, net_color, linestyles = (
    load_definitions()
)
t_deltas = {
    "jed0011": 0,
    "jed0022": 4,
    "jed0033": 2,
}
datasets = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/latlon/atm_2d_latlon.nc"
    ).pipe(fix_time)


# %% calculate iorg raw
a = 1.228
b = -1.106 * 1e-3
T_thresh = 273.15 - 38
sigma = 5.67e-8
T_eff = T_thresh * (a + b * T_thresh)
rad_thresh = sigma * T_eff**4

iorgs = {}
masks = {}
for run in runs:
    masks[run] = datasets[run].rlut < rad_thresh
    iorg = []
    for time in masks[run].time:
        iorg.append(
            cm.mask.iorg_objects(mask=masks[run].sel(time=time), periodic_domain=False)
        )
    iorgs[run] = xr.DataArray(iorg, coords={"time": masks[run].time}, dims=["time"])

# %% plot Iorgs raw with boxplot
fig, ax = plt.subplots(figsize=(4, 6))
for run in runs:
    bp = ax.boxplot(
        iorgs[run],
        label=line_labels[run],
        showfliers=False,
        positions=[t_deltas[run]],
        widths=0.5,
        patch_artist=True,
        boxprops={"facecolor": colors[run], "edgecolor": "black"},
        medianprops={"color": "black"},
    )
ax.set_xticklabels(["Control", "+2 K", "+4 K"])
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylabel("Iorg")
fig.savefig("plots/misc/iorg_raw_boxplot.pdf", bbox_inches="tight")

# %% calculate iorg local minima
iorgs_min = {}
masks_min = {}
loc_minima = {}
for run in runs:
    print(run)
    mask = datasets[run].rlut < rad_thresh
    loc_minima[run] = ndimage.minimum_filter(mask*datasets[run].rlut, size=3)
    masks_min[run] = mask * np.where(loc_minima[run] == datasets[run].rlut, 1, 0)
    iorg_min = []
    for time in masks_min[run].time:
        iorg_min.append(
            cm.mask.iorg_objects(
                mask=masks_min[run].sel(time=time), periodic_domain=False
            )
        )
    iorgs_min[run] = xr.DataArray(
        iorg_min, coords={"time": masks_min[run].time}, dims=["time"]
    )

# %% plot Iorgs local minima with boxplot
fig, ax = plt.subplots(figsize=(4, 6))
for run in runs:
    bp = ax.boxplot(
        iorgs_min[run],
        label=line_labels[run],
        showfliers=False,
        positions=[t_deltas[run]],
        widths=0.5,
        patch_artist=True,
        boxprops={"facecolor": colors[run], "edgecolor": "black"},
        medianprops={"color": "black"},
    )
ax.set_xticklabels(["Control", "+2 K", "+4 K"])
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylabel("Iorg local minima")

# %% plot maps of mask and local minima
fig, axes = plt.subplots(
    1, 3, figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()}
)

for i, run in enumerate(runs):
    ax = axes[i]
    ax.set_title(line_labels[run])
    factor = 1
    mask = masks[run].isel(time=-1).coarsen(lat=factor, lon=factor, boundary="trim").mean()
    mask_min = masks_min[run].isel(time=-1).coarsen(lat=factor, lon=factor, boundary="trim").mean()

    # Plot mask
    mask.sel(lat=slice(0,10), lon=slice(0,10)).plot.pcolormesh(
        ax=ax,
        x="lon",
        y="lat",
        cmap="Greys",
        add_colorbar=False,
        transform=ccrs.PlateCarree(),
    )
    ax.set_title(line_labels[run])

    # Plot local minima
    minima = mask_min.sel(lat=slice(0,10), lon=slice(0,10))
    yy, xx = np.meshgrid(minima['lat'], minima['lon'], indexing='ij')
    mask = minima.values > 0  # or == 1, depending on your mask

    # Plot only the minima as dots
    ax.scatter(
        xx[mask], yy[mask],
        color='red', s=0.1, transform=ccrs.PlateCarree(), label='Local minima', alpha=1
    )
fig.savefig("plots/misc/iorg_local_minima_maps.pdf", bbox_inches="tight")
# %%
