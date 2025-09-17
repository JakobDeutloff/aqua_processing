# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import easygems.healpix as egh
import numpy as np
from src.grid_helpers import merge_grid
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os

# %% load data
ds = xr.open_dataset("/work/bm1183/m301049/icon_hcap_data/control/production/twp_hp.nc")
ds.attrs = {
    "grid_file_uri": "http://icon-downloads.mpimet.mpg.de/grids/public/mpim/0015/icon_grid_0015_R02B09_G.nc"
}
ds = ds.pipe(merge_grid)


# %% make colormap that goes from steelblue to white without alpha channel
background = (0, 0, 0)
white_cmap = LinearSegmentedColormap.from_list("white", [background, "#FF7FBD"])

# %% setup figure
fig_height_in = 14
fig_width_in = 24.3
projection = ccrs.Orthographic()

# %% Precompute all frames in parallel and save as PNGs
output_dir = "/work/bm1183/m301049/animation/frames"
os.makedirs(output_dir, exist_ok=True)


def save_frame_png(frame):

    # prepare
    fig, ax = plt.subplots(
        figsize=(fig_width_in, fig_height_in), subplot_kw={"projection": projection}
    )
    fig.set_dpi(300)

    ax.set_facecolor(background)
    ax.spines["geo"].set_edgecolor(background)
    fig.patch.set_facecolor(background)
    ax.set_global()
    _, _, nx, ny = np.array(ax.bbox.bounds, dtype=int)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    # subsample
    arr = egh.healpix_resample(
        ds["twp"].isel(time=frame).values,
        xlims,
        ylims,
        nx,
        ny,
        ax.projection,
        "nearest",
        nest=True,
    )

    # plot
    ax.imshow(
        arr,
        extent=xlims + ylims,
        origin="lower",
        cmap=white_cmap,
        norm=LogNorm(5e-2, 1e1),
    )

    # Tighten layout
    fig.subplots_adjust(top=0.95, bottom=0.05)

    # save
    fname = os.path.join(output_dir, f"frame_{frame:04d}.png")
    fig.savefig(fname)
    plt.close(fig)
    return fname


with ProcessPoolExecutor(max_workers=64) as executor:
    list(tqdm(executor.map(save_frame_png, range(len(ds.time))), total=len(ds.time)))

# %%
anim_path = "/work/bm1183/m301049/animation/animation_pink_5.mp4"
if os.path.exists(anim_path):
    os.remove(anim_path)

os.system(
    f"ffmpeg -framerate 25 -i /work/bm1183/m301049/animation/frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {anim_path}"
)


# %%
