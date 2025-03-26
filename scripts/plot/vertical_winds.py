# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# %% load data
runs = ["jed0011", "jed0033", "jed0022"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
colors = {"jed0011": "k", "jed0022": "r", "jed0033": "orange"}
datasets = {}
hrs = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed_20.nc"
    ).sel(index=slice(0, 1e6))

vgrid = (
    xr.open_dataset(
        "/work/bm1183/m301049/icon-mpim/experiments/jed0001/atm_vgrid_angel.nc"
    )
    .mean("ncells")
    .rename({"height": "height_2", "height_2": "height"})
)
# %% bin wa by IWP
wa_binned = {}
iwp_bins = np.logspace(-4, 1, 51)
iwp_points = (iwp_bins[:-1] + iwp_bins[1:]) / 2
for run in runs:
    wa_binned[run] = (
        datasets[run]["wa"].groupby_bins(datasets[run]["iwp"], bins=iwp_bins).std()
    )

# %% calculate cf
cf = {}
cf_binned = {}
for run in runs:
    cf[run] = (
        (
            datasets[run]["cli"]
            + datasets[run]["clw"]
            + datasets[run]["qr"]
            + datasets[run]["qg"]
            + datasets[run]["qs"]
        )
        > 5e-7
    ).astype(int)
    cf_binned[run] = cf[run].groupby_bins(datasets[run]["iwp"], bins=iwp_bins).mean()

# %% plot wa and cf

fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True, sharex=True)
max_height = np.abs(vgrid["zg"] - 18e3).argmin("height").values
min_height = np.abs(vgrid["zg"] - 6e3).argmin("height").values
max_height_2 = np.abs(vgrid["zghalf"] - 18e3).argmin("height_2").values
min_height_2 = np.abs(vgrid["zghalf"] - 6e3).argmin("height_2").values

for i, run in enumerate(runs):
    w = axes[i].contourf(
        iwp_points,
        vgrid["zghalf"].isel(height_2=slice(max_height_2, min_height_2)).values / 1e3,
        wa_binned[run].isel(height_2=slice(max_height_2, min_height_2)).T,
        levels=np.linspace(0, 0.3, 21),
        extend="both",
        cmap="viridis",
    )
    contour = axes[i].contour(
        iwp_points,
        vgrid["zg"].isel(height=slice(max_height, min_height)).values / 1e3,
        cf_binned[run].isel(height=slice(max_height, min_height)).T,
        colors="k",
        levels=[0.1, 0.3, 0.5, 0.7, 0.9],
    )
    axes[i].clabel(contour, inline=True, fontsize=8, fmt="%1.1f")
    axes[i].set_xscale("log")
    axes[i].set_xlim([1e-4, 10])
    axes[i].spines[["top", "right"]].set_visible(False)
    axes[i].set_xlabel("$I$ / kg m$^{-2}$")
    axes[i].set_title(exp_name[run])

cb = fig.colorbar(mappable=w, ax=axes, orientation="horizontal", pad=0.15, aspect=50)
cb.set_ticks([-0.08, -0.04, 0, 0.04, 0.08])
cb.set_ticklabels([-0.08, -0.04, 0, 0.04, 0.08])
cb.set_label("Std vertical velocity / m s$^{-1}$")
fig.savefig("plots/iwp_drivers/std_wa.png", dpi=300, bbox_inches="tight")


# %%
