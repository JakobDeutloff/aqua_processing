# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os
import ccic

# %% load data
ds = xr.open_mfdataset("/work/bm1183/m301049/ccic/raw/*.zarr", engine="zarr")
ds = ds.sortby('latitude')

# %% make colormap that goes from steelblue to white without alpha channel
background = (1, 1, 1)
white_cmap = LinearSegmentedColormap.from_list("white", [background, "#034E23"])

# %% setup figure
fig_height_in = 14
fig_width_in = 24.3
projection = ccrs.Orthographic()

# %% Precompute all frames in parallel and save as PNGs
output_dir = "/work/bm1183/m301049/animation/frames_ccic"
os.makedirs(output_dir, exist_ok=True)


def save_frame_png(frame):

    
    fig, ax = plt.subplots(
            figsize=(fig_width_in, fig_height_in), subplot_kw={"projection": projection}
        )
    dpi = 300
    fig.set_dpi(dpi)

    ax.set_facecolor(background)
    ax.spines["geo"].set_edgecolor(background)
    fig.patch.set_facecolor(background)
    ax.set_global()

    # plot
    arr = ds['tiwp'].isel(time=frame).values
    ax.imshow(
        arr,
        extent=[-180, 180, -60, 60],
        origin="lower",
        cmap=white_cmap,
        norm=LogNorm(5e-2, 1e1),
        transform=ccrs.PlateCarree(),
        rasterized=True,
    )

    # Tighten layout
    fig.subplots_adjust(top=0.95, bottom=0.05)

    # save
    fname = os.path.join(output_dir, f"frame_{frame:04d}.png")
    fig.savefig(fname)
    plt.close(fig)

    return fname

# %%

with ProcessPoolExecutor(max_workers=32) as executor:
    list(tqdm(executor.map(save_frame_png, range(len(ds.time))), total=len(ds.time)))

# %%
anim_path = "/work/bm1183/m301049/animation/animation_ccic_with_green.mp4"
if os.path.exists(anim_path):
    os.remove(anim_path)

os.system(
    f"ffmpeg -framerate 25 -i /work/bm1183/m301049/animation/frames_ccic/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {anim_path}"
)

# %%
