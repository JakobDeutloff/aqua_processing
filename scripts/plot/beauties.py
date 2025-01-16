# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import easygems.healpix as egh
import numpy as np
from src.grid_helpers import to_healpix, merge_grid
from matplotlib.colors import LinearSegmentedColormap, LogNorm

# %% load data
ds = (
    xr.open_dataset(
        "/work/bm1183/m301049/icon-mpim/experiments/jed0011/jed0011_atm_2d_19790729T000000Z.14380341.nc",
        chunks={'time': 1},
    )
    .pipe(merge_grid)
    .chunk(dict(ncells=-1))
)

# %% regrid to healpix
ds_hp = to_healpix(ds)

# %% 
transparent_white_cmap = LinearSegmentedColormap.from_list(
    'transparent_white', [(1, 1, 1, 0), (1, 1, 1, 1)]
)

# %% load cloud condensate 
dummy = ds_hp.isel(time=0)
cond = (dummy['clivi'] + dummy['qsvi'] + dummy['qgvi'] + dummy['qrvi'] + dummy['cllvi']).load()

# %% plot cloud condensate

screen_width_pixels = 3024  # Replace with 3456 for 16-inch
screen_height_pixels = 1964  # Replace with 2234 for 16-inch
screen_ppi = 254  # PPI of the screen

# Compute figure size in inches
fig_width_in = screen_width_pixels / screen_ppi
fig_height_in = screen_height_pixels / screen_ppi

projection = ccrs.Mollweide()
fig, ax = plt.subplots(
    figsize=(fig_width_in, fig_height_in), subplot_kw={"projection": projection}, constrained_layout=True
)
fig.set_dpi(300)
steelblue = '#07222e'
ax.set_facecolor(steelblue)
ax.spines['geo'].set_edgecolor(steelblue)
fig.patch.set_facecolor(steelblue)
ax.set_global()

_, _, nx, ny = np.array(ax.bbox.bounds, dtype=int)

xlims = ax.get_xlim()
ylims = ax.get_ylim()
im = egh.healpix_resample(cond, xlims, ylims, nx, ny, ax.projection, 'nearest', nest=True)

ax.imshow(im, extent=xlims + ylims, origin="lower", cmap=transparent_white_cmap, norm=LogNorm(3e-2, 1e1))
fig.savefig('plots/screensaver.png', dpi=300)
# %%
