# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# %% load data
runs = ["jed0011", "jed0033", "jed0022"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
datasets = {}
hrs = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed.nc"
    ).sel(index=slice(0, 1e6))
    hrs[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_heating_rates.nc"
    ).sel(index=slice(0, 1e6))

vgrid = (
    xr.open_dataset(
        "/work/bm1183/m301049/icon-mpim/experiments/jed0001/atm_vgrid_angel.nc"
    )
    .mean("ncells")
    .rename({"height": "height_2", "height_2": "height"})
)
# %% bin hrs by iwp
hrs_binned_net = {}
hrs_binned_sw = {}
hrs_binned_lw = {}
masks_hc = {}
iwp_bins = np.logspace(-4, 1, 51)
iwp_points = (iwp_bins[:-1] + iwp_bins[1:]) / 2

for run in runs:
    hrs_binned_net[run] = (
        hrs[run]["net_hr"].groupby_bins(datasets[run]["iwp"], bins=iwp_bins).mean()
    )
    hrs_binned_sw[run] = (
        hrs[run]["sw_hr"].groupby_bins(datasets[run]["iwp"], bins=iwp_bins).mean()
    )
    hrs_binned_lw[run] = (
        hrs[run]["lw_hr"].groupby_bins(datasets[run]["iwp"], bins=iwp_bins).mean()
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

# %% look at mean heating rates over 6 km
fig, axes = plt.subplots(3, 3, figsize=(16, 16), sharex=True, sharey=True)

max_height = np.abs(vgrid["zg"] - 18e3).argmin("height").values
min_height = np.abs(vgrid["zg"] - 6e3).argmin("height").values

for i, run in enumerate(runs):
    net = axes[i, 0].pcolormesh(
        iwp_points,
        vgrid["zg"].isel(height=slice(max_height, min_height)).values / 1e3,
        hrs_binned_net[run].isel(height=slice(max_height, min_height)).T,
        cmap="seismic",
        vmin=-5,
        vmax=5,
    )

    lw = axes[i, 1].pcolormesh(
        iwp_points,
        vgrid["zg"].isel(height=slice(max_height, min_height)).values / 1e3,
        hrs_binned_lw[run].isel(height=slice(max_height, min_height)).T,
        cmap="seismic",
        vmin=-5,
        vmax=5,
    )

    sw = axes[i, 2].pcolormesh(
        iwp_points,
        vgrid["zg"].isel(height=slice(max_height, min_height)).values / 1e3,
        hrs_binned_sw[run].isel(height=slice(max_height, min_height)).T,
        cmap="Reds",
        vmin=0,
        vmax=5,
    )

    for ax in axes[i, :]:
        contour = ax.contour(
            iwp_points,
            vgrid["zg"].isel(height=slice(max_height, min_height)).values / 1e3,
            cf_binned[run].isel(height=slice(max_height, min_height)).T,
            colors="k",
            levels=[0.1, 0.3, 0.5, 0.7, 0.9],
        )
        ax.clabel(contour, inline=True, fontsize=8, fmt="%1.1f")


for ax in axes.flatten():
    ax.set_xscale("log")
    ax.set_xlim(iwp_bins[0], iwp_bins[-1])


fig.text(0.07, 0.77, "Control", fontsize=14, rotation=90)
fig.text(0.07, 0.56, "+2K", fontsize=14, rotation=90)
fig.text(0.07, 0.34, "+4K", fontsize=14, rotation=90)

for ax in axes[:, 0]:
    ax.set_ylabel("Height / km")

for ax in axes[2, :]:
    ax.set_xlabel("$I$ / kg m$^{-2}$")

cb_net = fig.colorbar(mappable=net, ax=axes[:, 0], orientation="horizontal", pad=0.05)
cb_net.set_label("Net heating rate / K day$^{-1}$")
cb_lw = fig.colorbar(mappable=lw, ax=axes[:, 1], orientation="horizontal", pad=0.05)
cb_lw.set_label("LW heating rate / K day$^{-1}$")
cb_sw = fig.colorbar(mappable=sw, ax=axes[:, 2], orientation="horizontal", pad=0.05)
cb_sw.set_label("SW heating rate / K day$^{-1}$")

fig.savefig("plots/iwp_drivers/heating_rates.png", dpi=300, bbox_inches="tight")


# %%
