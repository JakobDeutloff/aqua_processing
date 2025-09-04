# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import easygems.healpix as egh
import numpy as np
from src.grid_helpers import merge_grid
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import matplotlib.animation as animation
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# %% load data
ds = xr.open_dataset('/work/bm1183/m301049/icon_hcap_data/control/production/twp_hp.nc')
ds.attrs = {
    'grid_file_uri': 'http://icon-downloads.mpimet.mpg.de/grids/public/mpim/0015/icon_grid_0015_R02B09_G.nc'
}
ds = ds.pipe(merge_grid)


# %% 
transparent_white_cmap = LinearSegmentedColormap.from_list(
    'transparent_white', [(1, 1, 1, 0), (1, 1, 1, 1)]
)

#%% setup figure
fig_height_in = 33.1
fig_width_in = 46.8
projection = ccrs.Mollweide()

fig, ax = plt.subplots(
    figsize=(fig_width_in, fig_height_in), subplot_kw={"projection": projection}
)
fig.set_dpi(300)
steelblue = '#07222e'
ax.set_facecolor(steelblue)
ax.spines['geo'].set_edgecolor(steelblue)
fig.patch.set_facecolor(steelblue)
ax.set_global()
fig.subplots_adjust(left=0.03, right=0.97)
_, _, nx, ny = np.array(ax.bbox.bounds, dtype=int)
xlims = ax.get_xlim()
ylims = ax.get_ylim()

def preprocess_frame(frame):
    return egh.healpix_resample(
        ds['twp'].isel(time=frame).values, xlims, ylims, nx, ny, ax.projection, 'nearest', nest=True
    )

# %% Precompute all frames in parallel
with ProcessPoolExecutor(max_workers=16) as executor:
    frames = list(tqdm(executor.map(preprocess_frame, range(25))))

#%% animate
img = ax.imshow(frames[0], extent=xlims + ylims, origin="lower", cmap=transparent_white_cmap, norm=LogNorm(3e-2, 1e1))

def update(frame):
    img.set_data(frames[frame])
    return (img,)

ani = animation.FuncAnimation(
    fig, update, frames=len(frames), interval=40, blit=True
)

# %%
ani.save("plots/animation.gif", fps=25)
# %%
