# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import easygems.healpix as egh
import numpy as np
from src.grid_helpers import to_healpix, merge_grid
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.colors import to_rgba

# %%
names = [
    "papa",
    "oma",
    "elisa",
    "remor",
]
colors =  {"papa": "#00491C", "oma": "#2A0142", "elisa": "#000000", "remor": "#010E5D"}

files = {
    "papa": 'jed0011_atm_2d_19790701T000040Z.15356915.nc',
    "oma": 'jed0011_atm_2d_19790721T000040Z.15371960.nc',
    "elisa": 'jed0011_atm_2d_19790716T000040Z.15365936.nc',
    "remor": 'jed0011_atm_2d_19790725T000040Z.15371960.nc',
}


# %% regrid to healpix
def plot_christmas(name):
    ds = (
        (
            xr.open_dataset(
                f"/work/bm1183/m301049/icon-mpim/experiments/jed0011/{files[name]}",
                chunks={"time": 1, 'ncells':-1},
            ).pipe(merge_grid)
        )
        .isel(time=0)
    )

    ds_hp = to_healpix(ds)

    #  load cloud condensate
    cond = (
        ds_hp["clivi"] + ds_hp["qsvi"] + ds_hp["qgvi"] + ds_hp["qrvi"] + ds_hp["cllvi"]
    ).load()

    # plot cloud condensate
    fig_height_in = 7.87402
    fig_width_in = 7.87402

    projection = ccrs.Orthographic()
    fig, ax = plt.subplots(
        figsize=(fig_width_in, fig_height_in), subplot_kw={"projection": projection}
    )
    fig.set_dpi(400)
    # Set background to transparent or white
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_global()
    ax.axis("off")  # Turn off axes BEFORE plotting

    _, _, nx, ny = np.array(ax.bbox.bounds, dtype=int)

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    im = egh.healpix_resample(
        cond.values, xlims, ylims, nx, ny, ax.projection, "nearest", nest=True
    )
    im = im.fillna(0)
    vmin, vmax = 6e-2, 1e1
    norm = LogNorm(vmin=vmin, vmax=vmax)
    # Create RGBA array with constant color
    rgba = np.zeros((*im.shape, 4))
    base_color = to_rgba(colors[name])[:3]  # Get RGB components
    rgba[..., 0] = base_color[0]  # R
    rgba[..., 1] = base_color[1]  # G
    rgba[..., 2] = base_color[2]  # B
    rgba[..., 3] = norm(np.clip(im, vmin, vmax))  # Alpha varies on log scale

    ax.imshow(rgba, extent=xlims + ylims, origin="lower")

    fig.savefig(
        f"plots/screensaver/clouds_{name}.pdf",
        bbox_inches="tight",
        pad_inches=0.5,
        dpi=400,
    )
    plt.close(fig)


# %%
for name in names:
    im = plot_christmas(name)
# %%
