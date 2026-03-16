# %%
import matplotlib.pyplot as plt
import numpy as np
from src.calc_variables import (
    calc_heating_rates_t,
)
from src.read_data import load_random_datasets, load_definitions
import xarray as xr

# %%
runs, exp_name, colors, line_labels, sw_color, lw_color, net_color, linestyles = (
    load_definitions()
)
datasets = load_random_datasets(version="temp")
datasets_processed = load_random_datasets(version="processed")
for run in runs:
    # Assign all variables from ds to datasets if dim == index
    ds = datasets_processed[run].sel(index=slice(0, 1e6))
    datasets[run] = datasets[run].assign(
        **{var: ds[var] for var in ds.variables if ("index",) == ds[var].dims}
    )
# %% calculate heating rates and cf
hrs = {}
hrs_binned = {}

iwp_bins = np.logspace(-4, 1, 51)
iwp_points = (iwp_bins[:-1] + iwp_bins[1:]) / 2
for run in runs:
    print(run)
    hrs[run] = calc_heating_rates_t(
        datasets[run]["rho"],
        datasets[run]["rsd"] - datasets[run]["rsu"],
        datasets[run]["rld"] - datasets[run]["rlu"],
        datasets[run]["zg"],
    )
    hrs_binned[run] = (
        hrs[run]["net_hr"].groupby_bins(datasets[run]["iwp"], bins=iwp_bins).mean()
    )

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

# %% plot
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharex=True, sharey=True)
cmap = "seismic"
net = axes[0].pcolormesh(
    iwp_points,
    hrs_binned["jed0011"]["temp"],
    hrs_binned["jed0011"].T,
    cmap=cmap,
    vmin=-4,
    vmax=4,
    rasterized=True,
)

diff = axes[1].pcolormesh(
    iwp_points,
    hrs_binned["jed0033"]["temp"],
    (hrs_binned["jed0033"] - hrs_binned["jed0011"]).T,
    cmap=cmap,
    vmin=-1.5,
    vmax=1.5,
    rasterized=True,
)

axes[2].pcolormesh(
    iwp_points,
    hrs_binned["jed0022"]["temp"],
    (hrs_binned["jed0022"] - hrs_binned["jed0011"]).T,
    cmap=cmap,
    vmin=-1.5,
    vmax=1.5,
    rasterized=True,
)

for i, run in enumerate(runs):
    contour = axes[i].contour(
        iwp_points,
        datasets[run]["temp"],
        cf_binned[run].T,
        colors="k",
        levels=[0.1, 0.3, 0.5, 0.7, 0.9],
    )
    axes[i].clabel(contour, inline=True, fontsize=8, fmt="%1.1f")
    axes[i].set_xscale("log")
    axes[i].set_xlim(iwp_bins[0], iwp_bins[-1])
    axes[i].invert_yaxis()
    axes[i].set_ylim([260, 190])
    axes[i].set_xlabel("$I$ / kg m$^{-2}$")

axes[0].set_ylabel("Temperature / K")
axes[0].set_yticks([260, 230, 210, 190])
axes[0].set_title("Control")
axes[1].set_title("+2 K - Control")
axes[2].set_title("+4 K - Control")


cbar_height = 0.03
cbar_bottom = -0.09  # vertical position
cbar_left1 = 0.13  # left for first colorbar
cbar_left2 = 0.4  # left for second colorbar

cax1 = fig.add_axes([cbar_left1, cbar_bottom, 0.22, cbar_height])
cax2 = fig.add_axes([cbar_left2, cbar_bottom, 0.5, cbar_height])


cb_net = fig.colorbar(mappable=net, cax=cax1, orientation="horizontal")
cb_net.set_label("$Q$ / K day$^{-1}$")

cb_diff = fig.colorbar(mappable=diff, cax=cax2, orientation="horizontal")
cb_diff.set_label(r"$\Delta Q $ / K day$^{-1}$")

# add letters
for i, ax in enumerate(axes):
    ax.text(
        0.03,
        1.05,
        chr(97 + i),
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
    )

fig.savefig("plots/publication/heating_rates.pdf", bbox_inches="tight", dpi=300)
# %% calculate change in f from changes in rho and changes in flux convergence
f_cont = (
    (
        (datasets["jed0011"]["rsd"] - datasets["jed0011"]["rsu"])
        + (datasets["jed0011"]["rld"] - datasets["jed0011"]["rlu"])
    ).diff("temp")
    / datasets["jed0011"]['zg'].diff("temp")
    * 86400
)
f_cont_binned = f_cont.groupby_bins(datasets["jed0011"]["iwp"], bins=iwp_bins).mean()
f_4K = (
    (
        (datasets["jed0022"]["rsd"] - datasets["jed0022"]["rsu"])
        + (datasets["jed0022"]["rld"] - datasets["jed0022"]["rlu"])
    ).diff("temp")
    / datasets["jed0022"]['zg'].diff("temp")
    * 86400
)
f_4K_binned = f_4K.groupby_bins(datasets["jed0022"]["iwp"], bins=iwp_bins).mean()

dens_cont = 1/(datasets["jed0011"]["rho"] * 1004)
dens_cont_binned = dens_cont.groupby_bins(datasets["jed0011"]["iwp"], bins=iwp_bins).mean()
dens_4K = 1/(datasets["jed0022"]["rho"] * 1004)
dens_4K_binned = dens_4K.groupby_bins(datasets["jed0022"]["iwp"], bins=iwp_bins).mean()

diff_dens = (dens_4K_binned - dens_cont_binned) * f_cont_binned
diff_flux = (f_4K_binned - f_cont_binned) * dens_cont_binned

# %% plot heating rates difference for 4K experiment with constant rho and constant flux
fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharex=True, sharey=True)

axes[0].pcolormesh(
    iwp_points,
    hrs_binned["jed0022"]["temp"],
    (hrs_binned["jed0022"] - hrs_binned["jed0011"]).T,
    cmap='PiYG',
    vmin=-0.5,
    vmax=0.5,
    rasterized=True,
)

axes[1].pcolormesh(
    iwp_points,
    diff_flux["temp"],
    diff_flux.T,
    cmap='PiYG',
    vmin=-0.5,
    vmax=0.5,
    rasterized=True,
)

im = axes[2].pcolormesh(
    iwp_points,
    diff_dens["temp"],
    diff_dens.T,
    cmap='PiYG',
    vmin=-0.5,
    vmax=0.5,
    rasterized=True,
)


for ax in axes:
    ax.set_xscale("log")
    ax.set_xlim(iwp_bins[0], iwp_bins[-1])
    ax.invert_yaxis()
    ax.set_ylim([260, 190])
    ax.set_xlabel("$I$ / kg m$^{-2}$")

axes[0].set_ylabel("Temperature / K")
axes[0].set_title("Total Change")
axes[1].set_title("Change from Flux Convergence")
axes[2].set_title("Change from Density")

# label axes
for i, ax in enumerate(axes):
    ax.text(
        -0.06,
        1.03,
        chr(97 + i),
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
    )

cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.1, pad=-0.35, label=r"$\Delta Q$ / K day$^{-1}$", aspect=40)
cb.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
fig.tight_layout()
fig.savefig("plots/publication/heating_rates_diff_flux_density.pdf", bbox_inches="tight", dpi=300)

# %% save data
hr_net = xr.Dataset(
    {
        run: xr.DataArray(
            hrs_binned[run].values,
            coords={"iwp_points": iwp_points, "temp": hrs_binned[run]["temp"]},
            dims=["iwp_points", "temp"],
        )
        for run in runs
    }
)

cf = xr.Dataset(
    {
        run: xr.DataArray(
            cf_binned[run].values,
            coords={"iwp_points": iwp_points, "temp": cf_binned[run]["temp"]},
            dims=["iwp_points", "temp"],
        )
        for run in runs
    }
)


hr_net.to_netcdf(
    "/work/bm1183/m301049/icon_hcap_data/publication/heating_rates/hr_net.nc"
)
cf.to_netcdf("/work/bm1183/m301049/icon_hcap_data/publication/heating_rates/cf.nc")
diff_flux = xr.Dataset(
    {
        run: xr.DataArray(
            diff_flux.values,
            coords={"iwp_points": iwp_points, "temp": diff_flux["temp"]},
            dims=["iwp_points", "temp"],
        )
        for run in runs
    }
)
diff_dens = xr.Dataset(
    {
        run: xr.DataArray(
            diff_dens.values,
            coords={"iwp_points": iwp_points, "temp": diff_dens["temp"]},
            dims=["iwp_points", "temp"],
        )
        for run in runs
    }
)
diff_flux.to_netcdf(
    "/work/bm1183/m301049/icon_hcap_data/publication/heating_rates/diff_flux.nc"
)
diff_dens.to_netcdf(
    "/work/bm1183/m301049/icon_hcap_data/publication/heating_rates/diff_dens.nc"
)
# %%
