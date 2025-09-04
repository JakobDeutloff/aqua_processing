# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from src.read_data import load_random_datasets, load_cape_cin, load_vgrid, load_definitions

# %% load CRE data
runs, exp_name, colors, line_labels, sw_color, lw_color, net_color, linestyles = (
    load_definitions()
)
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
colors = {"jed0011": "k", "jed0022": "red", "jed0033": "orange"}
iwp_bins = np.logspace(-4, np.log10(40), 51)
datasets = load_random_datasets()
vgrid = load_vgrid()
ds_cape = load_cape_cin()


# %% difference in SW down for I>1
for run in runs:
    sw_down = datasets[run]["rsdt"].where(datasets[run]["iwp"] > 1).mean()
    print(f"{run} {sw_down.values}")

# %% bin quantities
iwp_bins = np.logspace(-4, np.log10(40), 31)
iwp_points = (iwp_bins[:-1] + iwp_bins[1:]) / 2
time_binned = {}
rad_time_binned = {}
sw_down_binned = {}
lat_binned = {}
time_std = {}
for run in runs:
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
        density=True,
    )
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
ax.set_title("Frequency of IWP > 1 kg m$^{-2}$")
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
ax.set_title("Frequency of Upward Velocities > 99th Percentile")
fig.tight_layout()
fig.savefig("plots/timing/daily_cycle_wa.png")

# %% fit sin to daily cycle of precip 
parameters_pr = {}

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
bins = np.arange(0, 25, 1)
x = (bins[:-1] + bins[1:]) / 2
for run in runs:
    hist, edges = np.histogram(
        datasets[run]["time_local"].where(datasets[run]['pr']> datasets[run]['pr'].quantile(0.99)),
        bins,
        density=True,
    )
    options = {"ftol": 1e-15}  # Adjust as needed
    popt, pcov = curve_fit(
        sin_func, x, hist, p0=[0.00517711, 1, 0.30754356], method="trf", **options
    )
    parameters_pr[run] = popt
    ax.stairs(hist, edges, label=exp_name[run], color=colors[run])
    ax.plot(bins, sin_func(bins, *popt), color=colors[run], linestyle="--")
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
ax.set_title("Frequency of Precipitation > 99th Percentile")
fig.tight_layout()
fig.savefig("plots/timing/daily_cycle_pr.png")

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

# %% compare phase of iwp and wa 
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
x = [1, 2, 3]
for i, run in enumerate(["jed0011", "jed0033", "jed0022"]):
    ax.scatter(
        x[i],
        6 - parameters_wa[run][1],
        color=colors[run],
        label=exp_name[run],
        marker="o"
    )
    ax.scatter(
        x[i],
        6 - parameters_iwp[run][1],
        color=colors[run],
        label=exp_name[run],
        marker="x"
    )
ax.spines[["top", "right"]].set_visible(False)
ax.set_xticks(x)
ax.set_xticklabels(['Control', '+2 K', '+4 K'])
ax.set_ylabel("Daily Maximum / h")
handles = [
    plt.Line2D([0], [0], color="grey", marker="o", linestyle=""),
    plt.Line2D([0], [0], color="grey", marker="x", linestyle=""),
]
labels = ["$W > P_{\mathrm{99}}(W)$", "IWP > 1 kg m$^{-2}$"]
ax.legend(handles, labels)


# %% look at CAPE and CIN vs IWP
iwp_bins = np.logspace(-4, np.log10(40), 31)
fig, axes = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
for run in runs:
    ds_cape[run]["cape"].groupby_bins(datasets[run]["iwp"], iwp_bins).mean().plot(
        ax=axes[0], label=exp_name[run], color=colors[run]
    )
    ds_cape[run]["cin"].groupby_bins(datasets[run]["iwp"], iwp_bins).mean().plot(
        ax=axes[1], label=exp_name[run], color=colors[run]
    )

axes[0].set_xscale("log")
# %% look at daily cycle of CAPE and CIN
fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
bins = np.arange(0, 25, 1)
for run in runs:

    axes[0].plot(
        bins[1:],
        ds_cape[run]["cape"]
        .groupby_bins(datasets[run]["time_local"], bins)
        .mean(),
        label=exp_name[run],
        color=colors[run],
    )
    axes[1].plot(
        bins[1:],
        ds_cape[run]["cin"]
        .groupby_bins(datasets[run]["time_local"], bins)
        .mean(),
        label=exp_name[run],
        color=colors[run],
    )
# %% plot upward velocity at 6 km binned by IWP
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
height = np.abs((vgrid["zghalf"] - 6e3)).argmin(dim="height_2").values
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


# %% bin fluxes 
latent_flux = {}
sensible_flux = {}
bins = np.arange(0, 25, 1)

for run in runs:
    mask = datasets[run]['iwp']<1e-2
    latent_flux[run] = (
        datasets[run]["ta"].sel(height=60).where(mask).groupby_bins(datasets[run]["time_local"], bins).mean()
    )
    sensible_flux[run] = (
        datasets[run]["hfss"].where(mask).groupby_bins(datasets[run]["time_local"], bins).mean()
    )


# %% plot fluxes
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
for run in runs:
    latent_flux[run].plot(
        ax=axes[0], label=exp_name[run], color=colors[run]
    )
    sensible_flux[run].plot(
        ax=axes[1], label=exp_name[run], color=colors[run]
    )   

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Local Time / h")

# %% look at lower troposphere temperature profiles
t_profile_day = {}
t_profile_night = {}
for run in runs:
    print(run)
    mask_day = (datasets[run]['time_local']> 12) & (datasets[run]['time_local'] < 16)
    mask_night = (datasets[run]['time_local']< 4)
    mask = (datasets[run]["iwp"] < 1e-2) & (datasets[run]['lwp']<1e-2)
    t_profile_day[run] = (
        datasets[run]["ta"].where(mask&mask_day).mean('index')
    )
    t_profile_night[run] = (
        datasets[run]["ta"].where(mask&mask_night).mean('index')
    )
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for run in runs:
    ax.plot(t_profile_day[run] - t_profile_night[run], vgrid['zg'], 
            label=exp_name[run], color=colors[run])
    
ax.set_ylim([0, 15e3])


# %%
