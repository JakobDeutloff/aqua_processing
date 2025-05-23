# %%
import xarray as xr
import matplotlib.pyplot as plt
from src.calc_variables import calc_cre
import numpy as np
from scipy.optimize import curve_fit
import metpy.calc as mpcalc
from metpy.units import units
from tqdm import tqdm

# %% load CRE data
runs = ["jed0011", "jed0033", "jed0022"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
colors = {"jed0011": "k", "jed0022": "red", "jed0033": "orange"}
cre_interp_mean = {}
cre_arr = {}
cre_arr_interp = {}
datasets = {}
for run in runs:
    cre_interp_mean[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/cre/{run}_cre_interp_raw.nc"
    )
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed.nc"
    )
    cape_cin = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_cape_cin.nc"
    )
    datasets[run] = datasets[run].assign(
        cape=cape_cin["cape"],
        cin=cape_cin["cin"],
    )
vgrid = (
    xr.open_dataset(
        "/work/bm1183/m301049/icon-mpim/experiments/jed0001/atm_vgrid_angel.nc"
    )
    .mean("ncells")
    .rename({"height_2": "height", "height": "height_2"})
)

# %% difference in SW down for I>1
for run in runs:
    sw_down = datasets[run]["rsdt"].where(datasets[run]["iwp"] > 1).mean()
    print(f"{run} {sw_down.values}")

# %% calculate cre clearsky and wetsky
for run in runs:
    cre_net, cre_sw, cre_lw = calc_cre(datasets[run], mode="clearsky")
    datasets[run] = datasets[run].assign(
        cre_net_cs=cre_net, cre_sw_cs=cre_sw, cre_lw_cs=cre_lw
    )
    cre_net, cre_sw, cre_lw = calc_cre(datasets[run], mode="wetsky")
    datasets[run] = datasets[run].assign(
        cre_net_ws=cre_net, cre_sw_ws=cre_sw, cre_lw_ws=cre_lw
    )
# %% assign local time
for run in runs:
    datasets[run] = datasets[run].assign(
        time_local=lambda d: d.time.dt.hour + (d.clon / 15)
    )
    datasets[run]["time_local"] = (
        datasets[run]["time_local"]
        .where(datasets[run]["time_local"] < 24, datasets[run]["time_local"] - 24)
        .where(datasets[run]["time_local"] > 0, datasets[run]["time_local"] + 24)
    )
    datasets[run]["time_local"].attrs = {"units": "h", "long_name": "Local time"}


# %% calculate cre high clouds
for run in runs:
    datasets[run] = datasets[run].assign(
        cre_net_hc=xr.where(
            datasets[run]["mask_low_cloud"],
            datasets[run]["cre_net_ws"],
            datasets[run]["cre_net_cs"],
        )
    )
    datasets[run]["cre_net_hc"].attrs = {
        "units": "W m^-2",
        "long_name": "Net High Cloud Radiative Effect",
    }
    datasets[run] = datasets[run].assign(
        cre_sw_hc=xr.where(
            datasets[run]["mask_low_cloud"],
            datasets[run]["cre_sw_ws"],
            datasets[run]["cre_sw_cs"],
        )
    )
    datasets[run]["cre_sw_hc"].attrs = {
        "units": "W m^-2",
        "long_name": "Shortwave High Cloud Radiative Effect",
    }
    datasets[run] = datasets[run].assign(
        cre_lw_hc=xr.where(
            datasets[run]["mask_low_cloud"],
            datasets[run]["cre_lw_ws"],
            datasets[run]["cre_lw_cs"],
        )
    )
    datasets[run]["cre_lw_hc"].attrs = {
        "units": "W m^-2",
        "long_name": "Longwave High Cloud Radiative Effect",
    }


# %%
iwp_bins = np.logspace(-4, np.log10(40), 31)
iwp_points = (iwp_bins[:-1] + iwp_bins[1:]) / 2
cre_net_hc_binned = {}
cre_sw_hc_binned = {}
cre_lw_hc_binned = {}
time_binned = {}
rad_time_binned = {}
sw_down_binned = {}
lat_binned = {}
time_std = {}
for run in runs:
    cre_net_hc_binned[run] = (
        datasets[run]["cre_net_hc"].groupby_bins(datasets[run]["iwp"], iwp_bins).mean()
    )
    cre_sw_hc_binned[run] = (
        datasets[run]["cre_sw_hc"].groupby_bins(datasets[run]["iwp"], iwp_bins).mean()
    )
    cre_lw_hc_binned[run] = (
        datasets[run]["cre_lw_hc"].groupby_bins(datasets[run]["iwp"], iwp_bins).mean()
    )
    time_binned[run] = (
        datasets[run]["time_local"].groupby_bins(datasets[run]["iwp"], iwp_bins).mean()
    )
    sw_down_binned[run] = (
        datasets[run]["rsdt"].groupby_bins(datasets[run]["iwp"], iwp_bins).mean()
    )
    lat_binned[run] = (
        np.abs(datasets[run]["clat"])
        .groupby_bins(datasets[run]["iwp"], iwp_bins)
        .mean()
    )
    rad_time_binned[run] = (
        np.abs(datasets[run]["time_local"] - 12)
        .groupby_bins(datasets[run]["iwp"], iwp_bins)
        .mean()
    )

# %% plot cre net wit and without interpolation
fig, ax = plt.subplots()

cre_interp_mean["jed0011"]["net"].plot(
    ax=ax, label="interpolated", color="k", linestyle="-"
)
cre_net_hc_binned["jed0011"].plot(ax=ax, label="binned", color="k", linestyle="--")
cre_interp_mean["jed0011"]["sw"].plot(
    ax=ax, label="interpolated", color="blue", linestyle="-"
)
cre_sw_hc_binned["jed0011"].plot(ax=ax, label="binned", color="blue", linestyle="--")
cre_interp_mean["jed0011"]["lw"].plot(
    ax=ax, label="interpolated", color="red", linestyle="-"
)
cre_lw_hc_binned["jed0011"].plot(ax=ax, label="binned", color="red", linestyle="--")

ax.set_xscale("log")
ax.set_xlabel("$I$ / $kg m^{-2}$")
ax.set_ylabel("$C(I)$ / $W m^{-2}$")
ax.set_xlim([1e-4, 40])
ax.spines[["top", "right"]].set_visible(False)

labels = ["Daily Mean CRE", "Real CRE"]
handles = [
    plt.Line2D([0], [0], color="grey", linestyle="-"),
    plt.Line2D([0], [0], color="grey", linestyle="--"),
]
ax.legend(handles, labels)
fig.savefig("plots/cre/cre_interp_vs_binned.png")


# %% compare CRE
linestyles = {
    "jed0011": "-",
    "jed0022": "--",
    "jed0033": ":",
}
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for run in runs:
    cre_net_hc_binned[run].plot(
        ax=ax,
        label=exp_name[run],
        color="k",
        linestyle=linestyles[run],
        alpha=0.5,
    )
    cre_sw_hc_binned[run].plot(
        ax=ax,
        label=exp_name[run],
        color="blue",
        linestyle=linestyles[run],
        alpha=0.5,
    )
    cre_lw_hc_binned[run].plot(
        ax=ax,
        label=exp_name[run],
        color="red",
        linestyle=linestyles[run],
        alpha=0.5,
    )
ax.set_xscale("log")
ax.set_xlim([1e-4, 10])

# %% plot time local
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
datasets["jed0011"]["time_local"].groupby_bins(
    datasets["jed0011"]["iwp"], iwp_bins
).mean().sel(iwp_bins=slice(1e-4, 10)).plot(ax=ax, color="k")

ax.set_xscale("log")
ax.set_xlabel("$I$ / $kg m^{-2}$")
ax.set_ylabel("Time Local / h")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig("plots/cre/time_local.png")

# %% plot mean time and SW down
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)


for run in runs:
    sw_down_binned[run].sel(iwp_bins=slice(1e-4, 10)).plot(
        ax=axes[0], label=exp_name[run], color=colors[run]
    )
    rad_time_binned[run].sel(iwp_bins=slice(1e-4, 10)).plot(
        ax=axes[1], label=exp_name[run], color=colors[run]
    )
    lat_binned[run].sel(iwp_bins=slice(1e-4, 10)).plot(
        ax=axes[2], label=exp_name[run], color=colors[run]
    )

axes[0].set_ylabel("SW down / W m$^{-2}$")
axes[1].set_ylabel("Time Difference to Noon / h")
axes[2].set_ylabel("Distance to equator / deg")
axes[0].set_xscale("log")
for ax in axes:
    ax.set_xlabel("$I$ / $kg m^{-2}$")
    ax.spines[["top", "right"]].set_visible(False)

fig.savefig("plots/cre/timing.png")

# %% plot diff in time and dist to eq
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

for run in runs[1:]:
    (
        sw_down_binned[run].sel(iwp_bins=slice(1e-4, 10))
        - sw_down_binned["jed0011"].sel(iwp_bins=slice(1e-4, 10))
    ).plot(ax=axes[0], label=exp_name[run], color=colors[run])
    (
        rad_time_binned[run].sel(iwp_bins=slice(1e-4, 10))
        - rad_time_binned["jed0011"].sel(iwp_bins=slice(1e-4, 10))
    ).plot(ax=axes[1], label=exp_name[run], color=colors[run])
    (
        lat_binned[run].sel(iwp_bins=slice(1e-4, 10))
        - lat_binned["jed0011"].sel(iwp_bins=slice(1e-4, 10))
    ).plot(ax=axes[2], label=exp_name[run], color=colors[run])

axes[0].set_ylabel("SW down / W m$^{-2}$")
axes[1].set_ylabel("Time Difference to Noon / h")
axes[2].set_ylabel("Distance to equator / deg")
axes[0].set_xscale("log")
for ax in axes:
    ax.set_xlabel("$I$ / $kg m^{-2}$")
    ax.spines[["top", "right"]].set_visible(False)

fig.savefig("plots/cre/timing_diff.png")

# %% plot dist of time_local for high IWP
for run in runs:
    hist, edges = np.histogram(
        datasets[run]["time_local"].where(datasets[run]["iwp"] > 1e0),
        np.arange(0, 25, 1),
        density=False,
    )
    hist = hist / hist.sum()
    ax.stairs(
        hist,
        edges,
        label=exp_name[run],
        color=colors[run],
        alpha=0.5,
    )


# %% fit sin to daily cycle
def sin_func(x, a, b, c):
    return a * np.sin((x * 2 * np.pi / 24) + (b * 2 * np.pi / 24)) + c


parameters_iwp = {}

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
bins = np.arange(0, 25, 1)
x = (bins[:-1] + bins[1:]) / 2
for run in runs:
    hist, edges = np.histogram(
        datasets[run]["time_local"].where(datasets[run]["iwp"] > 1e0),
        bins,
        density=False,
    )
    hist = hist / hist.sum()
    options = {"ftol": 1e-15}  # Adjust as needed
    popt, pcov = curve_fit(
        sin_func, x, hist, p0=[0.00517711, 1, 0.30754356], method="trf", **options
    )
    parameters_iwp[run] = popt
    ax.stairs(hist, edges, label=exp_name[run], color=colors[run])
    ax.plot(bins, sin_func(bins, *popt), color=colors[run], linestyle="--")
ax.set_xlim([0.1, 23.9])
ax.set_ylim([0.03, 0.055])
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("Local Time / h")
ax.set_ylabel("Probability")
labels = ["Control", "+2 K", "+4 K", "Sinus Fit", "Histogram"]
handles = [
    plt.Line2D([0], [0], color="k", linestyle="-"),
    plt.Line2D([0], [0], color="orange", linestyle="-"),
    plt.Line2D([0], [0], color="red", linestyle="-"),
    plt.Line2D([0], [0], color="grey", linestyle="--"),
    plt.Line2D([0], [0], color="grey", linestyle="-"),
]
ax.legend(handles, labels)
fig.tight_layout()
fig.savefig("plots/timing/daily_cycle.png")

# %% histogram of local time for wa > 1
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
bins = np.arange(0, 25, 1)
quantiles = {}
x = (bins[:-1] + bins[1:]) / 2
height_1 = np.abs((vgrid["zghalf"] - 2e3)).argmin(dim="height_2").values
height_2 = np.abs((vgrid["zghalf"] - 15e3)).argmin(dim="height_2").values
parameters_wa = {}
for run in runs:
    vels = (
        datasets[run]["wa"].isel(height_2=slice(height_2, height_1)).max(dim="height_2")
    )
    quantiles[run] = vels.quantile(0.99)
    mask = vels > quantiles[run]

    hist, edges = np.histogram(
        datasets[run]["time_local"].where(mask),
        bins,
        density=True,
    )

    # fit sin

    popt, pcov = curve_fit(
        sin_func, x, hist, p0=[0.00517711, 1, 0.30754356], method="trf", **options
    )
    parameters_wa[run] = popt

    ax.stairs(
        hist,
        edges,
        label=exp_name[run],
        color=colors[run],
    )
    ax.plot(x, sin_func(x, *popt), color=colors[run], linestyle="--")
ax.set_xlim([0.1, 23.9])
ax.set_ylim([0.03, 0.06])
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("Local Time / h")
ax.set_ylabel("Probability")
labels = ["Control", "+2 K", "+4 K", "Sinus Fit", "Histogram"]
handles = [
    plt.Line2D([0], [0], color="k", linestyle="-"),
    plt.Line2D([0], [0], color="orange", linestyle="-"),
    plt.Line2D([0], [0], color="red", linestyle="-"),
    plt.Line2D([0], [0], color="grey", linestyle="--"),
    plt.Line2D([0], [0], color="grey", linestyle="-"),
]
ax.legend(handles, labels)
fig.tight_layout()
fig.savefig("plots/timing/daily_cycle_wa.png")

# %% plot correlation of iwp and wa
fig, axes = plt.subplots(1, 3, figsize=(10, 5))
for i, run in enumerate(runs):
    mask = datasets[run]["iwp"] > 1e0
    axes[i].scatter(
        datasets[run]["iwp"].where(mask),
        datasets[run]["wa"].where(mask).isel(height_2=height),
        color=colors[run],
        s=0.1,
        alpha=0.5,
    )
    axes[i].set_xscale("log")


# %% plot phase and amplitude of iwp fits
fig, axes = plt.subplots(2, 1, figsize=(4, 5))
T_delta = {
    "jed0011": 0,
    "jed0022": 4,
    "jed0033": 2,
}
for run in ["jed0022", "jed0033", "jed0011"]:
    axes[0].scatter(
        T_delta[run],
        parameters_iwp[run][1],
        color=colors[run],
    )
    axes[1].scatter(
        T_delta[run],
        parameters_iwp[run][0],
        color=colors[run],
    )

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([0, 2, 4])
    ax.set_xticklabels([0, 2, 4])


axes[0].set_ylabel("Phase / h")
axes[1].set_ylabel("Amplitude")
axes[1].set_xlabel("Temperature Change / K")
fig.tight_layout()
fig.savefig("plots/timing/daily_cycle_fit.png")

# %% compare phase of iwp ans wa


# %% look at CAPE and CIN vs IWP
iwp_bins = np.logspace(-4, np.log10(40), 31)
fig, axes = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
for run in runs:
    datasets[run]["cape"].groupby_bins(datasets[run]["iwp"], iwp_bins).mean().plot(
        ax=axes[0], label=exp_name[run], color=colors[run]
    )
    datasets[run]["cin"].groupby_bins(datasets[run]["iwp"], iwp_bins).mean().plot(
        ax=axes[1], label=exp_name[run], color=colors[run]
    )

axes[0].set_xscale("log")
# %% look at daily cycle of CAPE and CIN
fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
bins = np.arange(0, 25, 1)
for run in runs:
    mask = (datasets[run]["iwp"] < 1e-4) & (datasets[run]["lwp"] < 1e-4)

    axes[0].plot(
        bins[1:],
        datasets[run]["cape"]
        .where(mask)
        .groupby_bins(datasets[run]["time_local"], bins)
        .mean(),
        label=exp_name[run],
        color=colors[run],
    )
    axes[1].plot(
        bins[1:],
        datasets[run]["cin"]
        .where(mask)
        .groupby_bins(datasets[run]["time_local"], bins)
        .mean(),
        label=exp_name[run],
        color=colors[run],
    )
# %% plot upward velocity at 6 km binned by IWP
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

for run in runs:
    datasets[run]["wa"].isel(height_2=height).groupby_bins(
        datasets[run]["iwp"], iwp_bins
    ).mean().plot(ax=ax, label=exp_name[run], color=colors[run])

ax.set_xscale("log")
# %% plot histogram uf upward velocities for IWP > 1
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
bins = np.arange(-2, 5, 0.1)
for run in runs:
    hist, edges = np.histogram(
        datasets[run]["wa"].where(datasets[run]["iwp"] > 10).isel(height_2=height),
        bins,
        density=False,
    )

    ax.stairs(
        hist,
        edges,
        label=exp_name[run],
        color=colors[run],
        alpha=0.5,
    )

# %% plot histogram of upward velocities above 1
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
bins = np.arange(1, 5, 0.1)
for run in runs:
    hist, edges = np.histogram(
        datasets[run]["wa"].isel(height_2=height),
        bins,
        density=False,
    )
    ax.stairs(
        hist,
        edges,
        label=exp_name[run],
        color=colors[run],
        alpha=0.5,
    )


# %%
