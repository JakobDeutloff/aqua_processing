# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from src.calc_variables import (
    calc_heating_rates_t,
)
# %%
runs = ["jed0011", "jed0033", "jed0022"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
colors = {"jed0011": "k", "jed0022": "r", "jed0033": "orange"}
datasets = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_tgrid_20.nc"
    )
    ds = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed.nc"
    ).sel(index=slice(0, 1e6))
    # Assign all variables from ds to datasets if dim == index
    datasets[run] = datasets[run].assign(
        **{var: ds[var] for var in ds.variables if ("index",) == ds[var].dims}
    )
# %% calculate heating rates and cf 
hrs = {}
hrs_binned_net = {}
hrs_binned_sw = {}
hrs_binned_lw = {}
iwp_bins = np.logspace(-6, 1, 51)
iwp_points = (iwp_bins[:-1] + iwp_bins[1:]) / 2
for run in runs:
    print(run)
    hrs[run] = calc_heating_rates_t(
        datasets[run]["rho"],
        datasets[run]["rsd"] - datasets[run]["rsu"],
        datasets[run]["rld"] - datasets[run]["rlu"],
        datasets[run]["zg"],
    )
    hrs_binned_net[run] = (
        hrs[run]["net_hr"].groupby_bins(datasets[run]["iwp"], bins=iwp_bins).mean()
    )
    hrs_binned_sw[run] = (
        hrs[run]["sw_hr"].groupby_bins(datasets[run]["iwp"], bins=iwp_bins).mean()
    )
    hrs_binned_lw[run] = (
        hrs[run]["lw_hr"].groupby_bins(datasets[run]["iwp"], bins=iwp_bins).mean()
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

# %% calculate binned hr and cf for day and night 
hrs_binned_lw_day = {}
hrs_binned_lw_night = {}
hrs_binned_sw_day = {}
hrs_binned_sw_night = {}
hrs_binned_net_day = {}
hrs_binned_net_night = {}
cf_binned_day = {}
cf_binned_night = {}
for run in runs:
    # assign local time
    datasets[run] = datasets[run].assign(time_local=lambda d: d.time.dt.hour + (d.clon / 15))
    datasets[run]["time_local"] = datasets[run]["time_local"].where(
        datasets[run]["time_local"] < 24, datasets[run]["time_local"] - 24
    ).where(
        datasets[run]["time_local"] > 0, datasets[run]["time_local"] + 24
    )
    datasets[run]["time_local"].attrs = {"units": "h", "long_name": "Local time"}

    # calculate binned hr and cf for day and night
    mask_day = (datasets[run]["time_local"] > 12) & (datasets[run]["time_local"] < 16)
    mask_night = (datasets[run]["time_local"] < 4)
    hrs_binned_net_day[run] = (
        hrs[run]["net_hr"]
        .where(mask_day)
        .groupby_bins(datasets[run]["iwp"], bins=iwp_bins)
        .mean()
    )
    hrs_binned_net_night[run] = (
        hrs[run]["net_hr"]
        .where(mask_night)
        .groupby_bins(datasets[run]["iwp"], bins=iwp_bins)
        .mean()
    )
    hrs_binned_sw_day[run] = (
        hrs[run]["sw_hr"]
        .where(mask_day)
        .groupby_bins(datasets[run]["iwp"], bins=iwp_bins)
        .mean()
        )
    hrs_binned_sw_night[run] = (
        hrs[run]["sw_hr"]
        .where(mask_night)
        .groupby_bins(datasets[run]["iwp"], bins=iwp_bins)
        .mean()
    )
    hrs_binned_lw_day[run] = (
        hrs[run]["lw_hr"]
        .where(mask_day)
        .groupby_bins(datasets[run]["iwp"], bins=iwp_bins)
        .mean()
    )
    hrs_binned_lw_night[run] = (
        hrs[run]["lw_hr"]
        .where(mask_night)
        .groupby_bins(datasets[run]["iwp"], bins=iwp_bins)
        .mean()
    )
    cf_binned_day[run] = (
        cf[run]
        .where(mask_day)
        .groupby_bins(datasets[run]["iwp"], bins=iwp_bins)
        .mean()
    )
    cf_binned_night[run] = (
        cf[run]
        .where(mask_night)
        .groupby_bins(datasets[run]["iwp"], bins=iwp_bins)
        .mean()
    )     


# %% define plotting functions

def plot_heatingrates(hrs_binned_net, hrs_binned_lw, hrs_binned_sw, cf_binned):
    fig, axes = plt.subplots(3, 3, figsize=(16, 16), sharex=True, sharey=True)

    for i, run in enumerate(runs):
        net = axes[i, 0].pcolormesh(
            iwp_points,
            hrs_binned_net[run]['temp'],
            hrs_binned_net[run].T,
            cmap="seismic",
            vmin=-5,
            vmax=5,
        )

        lw = axes[i, 1].pcolormesh(
            iwp_points,
            hrs_binned_lw[run]['temp'],
            hrs_binned_lw[run].T,
            cmap="seismic",
            vmin=-5,
            vmax=5,
        )

        sw = axes[i, 2].pcolormesh(
            iwp_points,
            hrs_binned_sw[run]['temp'],
            hrs_binned_sw[run].T,
            cmap="Reds",
            vmin=0,
            vmax=5,
        )

        for ax in axes[i, :]:
            contour = ax.contour(
                iwp_points,
                datasets[run]['temp'],
                cf_binned[run].T,
                colors="k",
                levels=[0.1, 0.3, 0.5, 0.7, 0.9],
            )
            ax.clabel(contour, inline=True, fontsize=8, fmt="%1.1f")


    for ax in axes.flatten():
        ax.set_xscale("log")
        ax.set_xlim(iwp_bins[0], iwp_bins[-1])
        ax.invert_yaxis()
        ax.set_ylim([260, 190])


    fig.text(0.07, 0.77, "Control", fontsize=14, rotation=90)
    fig.text(0.07, 0.56, "+2K", fontsize=14, rotation=90)
    fig.text(0.07, 0.34, "+4K", fontsize=14, rotation=90)

    for ax in axes[:, 0]:
        ax.set_ylabel("Temperature / K")

    for ax in axes[2, :]:
        ax.set_xlabel("$I$ / kg m$^{-2}$")

    cb_net = fig.colorbar(mappable=net, ax=axes[:, 0], orientation="horizontal", pad=0.05)
    cb_net.set_label("Net heating rate / K day$^{-1}$")
    cb_lw = fig.colorbar(mappable=lw, ax=axes[:, 1], orientation="horizontal", pad=0.05)
    cb_lw.set_label("LW heating rate / K day$^{-1}$")
    cb_sw = fig.colorbar(mappable=sw, ax=axes[:, 2], orientation="horizontal", pad=0.05)
    cb_sw.set_label("SW heating rate / K day$^{-1}$")
    return fig, axes

def plot_diff_heatingrates(hrs_binned_net, hrs_binned_lw, hrs_binned_sw, cf_binned):
    fig, axes = plt.subplots(2, 3, figsize=(16, 12), sharex=True, sharey=True)

    for i, run in enumerate(runs[1:]):

        net = axes[i, 0].pcolormesh(
            iwp_points,
            hrs_binned_net[run]['temp'],
            (hrs_binned_net[run] - hrs_binned_net['jed0011']).T,
            cmap="seismic",
            vmin=-1.5,
            vmax=1.5,
        )
        lw = axes[i, 1].pcolormesh(
            iwp_points,
            hrs_binned_lw[run]['temp'],
            (hrs_binned_lw[run] - hrs_binned_lw['jed0011']).T,
            cmap="seismic",
            vmin=-1.5,
            vmax=1.5,
        )
        sw = axes[i, 2].pcolormesh(
            iwp_points,
            hrs_binned_sw[run]['temp'],
            (hrs_binned_sw[run] - hrs_binned_sw['jed0011']).T,
            cmap="seismic",
            vmin=-1.5,
            vmax=1.5,
        )
        for ax in axes[i, :]:
            contour = ax.contour(
                iwp_points,
                datasets[run]['temp'],
                cf_binned[run].T,
                colors="k",
                levels=[0.1, 0.3, 0.5, 0.7, 0.9],
            )
            ax.clabel(contour, inline=True, fontsize=8, fmt="%1.1f")

    for ax in axes.flatten():
        ax.set_xscale("log")
        ax.set_xlim(iwp_bins[0], iwp_bins[-1])
        ax.invert_yaxis()
        ax.set_ylim([260, 190])

    for ax in axes[:, 0]:
        ax.set_ylabel("Temperature / K")

    for ax in axes[1, :]:
        ax.set_xlabel("$I$ / kg m$^{-2}$")

    fig.text(0.07, 0.68, "+2K - Control", fontsize=14, rotation=90)
    fig.text(0.07, 0.33, "+4K - Control", fontsize=14, rotation=90)

    cb_net = fig.colorbar(mappable=net, ax=axes[:, 0], orientation="horizontal", pad=0.08)
    cb_net.set_label("Net heating rate / K day$^{-1}$")
    cb_lw = fig.colorbar(mappable=lw, ax=axes[:, 1], orientation="horizontal", pad=0.08)
    cb_lw.set_label("LW heating rate / K day$^{-1}$")
    cb_sw = fig.colorbar(mappable=sw, ax=axes[:, 2], orientation="horizontal", pad=0.08)
    cb_sw.set_label("SW heating rate / K day$^{-1}$")

    return fig, axes


# %% plot total heatingrates
fig, axes = plot_heatingrates(hrs_binned_net, hrs_binned_lw, hrs_binned_sw, cf_binned)
fig.savefig(
    "plots/heating_rates/heating_rates_temp_all.png", dpi=300, bbox_inches="tight")

fig, axes = plot_heatingrates(hrs_binned_net_day, hrs_binned_lw_day, hrs_binned_sw_day, cf_binned_day)
fig.savefig(
    "plots/heating_rates/heating_rates_temp_day.png", dpi=300, bbox_inches="tight"
)
fig, axes = plot_heatingrates(hrs_binned_net_night, hrs_binned_lw_night, hrs_binned_sw_night, cf_binned_night)
fig.savefig(
    "plots/heating_rates/heating_rates_temp_night.png", dpi=300, bbox_inches="tight"
)


# %% plot diff heatingrates
fig, axes = plot_diff_heatingrates(hrs_binned_net, hrs_binned_lw, hrs_binned_sw, cf_binned)
fig.savefig(
    "plots/heating_rates/heating_rates_temp_diff_all.png", dpi=300, bbox_inches="tight"
)
fig, axes = plot_diff_heatingrates(hrs_binned_net_day, hrs_binned_lw_day, hrs_binned_sw_day, cf_binned_day)
fig.savefig(
    "plots/heating_rates/heating_rates_temp_diff_day.png", dpi=300, bbox_inches="tight"
)
fig, axes = plot_diff_heatingrates(hrs_binned_net_night, hrs_binned_lw_night, hrs_binned_sw_night, cf_binned_night)
fig.savefig(
    "plots/heating_rates/heating_rates_temp_diff_night.png", dpi=300, bbox_inches="tight"
)

# %% differential heating dry and moist 

time_bins = np.arange(0, 25, 1)
hrs_binned_net_dry = {}
hrs_binned_net_moist = {}

for run in runs: 
    mask_dry = datasets[run]['iwp'] < 1e-4
    mask_moist = datasets[run]['iwp'] > 1
    hrs_binned_net_dry[run] = (
        hrs[run]['net_hr']
        .where(mask_dry)
        .groupby_bins(datasets[run]['time_local'], bins=time_bins)
        .mean()
    )
    hrs_binned_net_moist[run] = (
        hrs[run]['net_hr']
        .where(mask_moist)
        .groupby_bins(datasets[run]['time_local'], bins=time_bins)
        .mean()
    )



# %% plot differential heating dry and moist
fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=True)


dry = axes[0, 0].pcolormesh(
    (time_bins[:-1] + time_bins[1:]) / 2,
    hrs_binned_net_dry['jed0011']['temp'],
    hrs_binned_net_dry['jed0011'].T,
    cmap="seismic",
    vmin=-2,
    vmax=2,
)
moist = axes[0, 1].pcolormesh(
    (time_bins[:-1] + time_bins[1:]) / 2,
    hrs_binned_net_moist['jed0011']['temp'],
    hrs_binned_net_moist['jed0011'].T,
    cmap="seismic",
    vmin=-2,
    vmax=2,
)
diff = axes[0, 2].pcolormesh(
    (time_bins[:-1] + time_bins[1:]) / 2,
    hrs_binned_net_moist['jed0011']['temp'],
    (hrs_binned_net_moist['jed0011'] - hrs_binned_net_dry['jed0011']).T,
    cmap="seismic",
    vmin=-2,
    vmax=2,
)

for i, run in enumerate(runs[1:]):

    dry = axes[i+1, 0].pcolormesh(
        (time_bins[:-1] + time_bins[1:]) / 2,
        hrs_binned_net_dry[run]['temp'],
        (hrs_binned_net_dry[run] - hrs_binned_net_dry['jed0011']).T,
        cmap="seismic",
        vmin=-1,
        vmax=1,
    )
    moist = axes[i+1, 1].pcolormesh(
        (time_bins[:-1] + time_bins[1:]) / 2,
        hrs_binned_net_moist[run]['temp'],
        (hrs_binned_net_moist[run] - hrs_binned_net_moist['jed0011']).T,
        cmap="seismic",
        vmin=-1,
        vmax=1,
    )
    diff = axes[i+1, 2].pcolormesh(
        (time_bins[:-1] + time_bins[1:]) / 2,
        hrs_binned_net_moist[run]['temp'],
        ((hrs_binned_net_moist[run] - hrs_binned_net_dry[run]) - (hrs_binned_net_moist['jed0011'] - hrs_binned_net_dry['jed0011'])).T,
        cmap="seismic",
        vmin=-1,
        vmax=1,
    )
    
for ax in axes.flatten():
    ax.set_xlim(0, 24)
    ax.invert_yaxis()
    ax.set_ylim([260, 190])
    

# %%
