# %%
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

# %% load data
runs = ["jed0011", "jed0022", "jed0033"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
datasets = {}
cre_interp_mean = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed.nc"
    )
    cre_interp_mean[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/cre/{run}_cre_interp_mean_rand_raw.nc"
    )

# %% calculate masks
mode = "temp_narrow"
iwp_bins = np.logspace(-4, np.log10(40), 51)
colors = {"jed0011": "k", "jed0022": "red", "jed0033": "orange"}
masks_height = {}
for run in runs:
    if mode == "temperature":
        masks_height[run] = datasets[run]["hc_top_temperature"] < (273.15 - 35)
    elif mode == "pressure":
        masks_height[run] = datasets[run]["hc_top_pressure"] < 350
    elif mode == "raw":
        masks_height[run] = True
    elif mode == "temp_narrow":
        masks_height[run] = (
            (datasets[run]["hc_top_temperature"] < (273.15 - 35))
            & (datasets[run]["clat"] < 20)
            & (datasets[run]["clat"] > -20)
        )
# %% investigate low cloud fraction
fig, ax = plt.subplots()

for run in runs:
    datasets[run]["mask_low_cloud"].groupby_bins(
        datasets[run]["iwp"], iwp_bins
    ).mean().plot(label=exp_name[run], color=colors[run])

ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("$I$  / kg m$^{-2}$")
ax.set_ylabel(r"$f_{\mathrm{lc}}(I)$")
ax.set_xscale("log")
ax.set_xlim([1e-4, 4e1])
ax.legend()
fig.savefig(
    f"plots/feedback/{mode}/low_cloud_fraction.png", dpi=300, bbox_inches="tight"
)

# %% investigate high cloud temperature and pressure
fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

for run in runs:
    datasets[run]["hc_top_temperature"].where(masks_height[run]).groupby_bins(
        datasets[run]["iwp"], iwp_bins
    ).mean().plot(ax=axes[0], label=exp_name[run], color=colors[run])
    datasets[run]["hc_top_pressure"].where(masks_height[run]).groupby_bins(
        datasets[run]["iwp"], iwp_bins
    ).mean().plot(ax=axes[1], label=exp_name[run], color=colors[run])


for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim([1e-4, 4e1])


axes[0].set_ylabel(r"$T_{\mathrm{hc}}$  / K")
axes[1].set_ylabel(r"$p_{\mathrm{hc}}$  / hPa")
axes[1].set_xlabel("$I$  / kg m$^{-2}$")
axes[0].set_xlabel("")
axes[0].set_xscale("log")

axes[0].legend()
fig.savefig(f"plots/feedback/{mode}/hc_top.png", dpi=300, bbox_inches="tight")

# %% plot 2d hist of high cloud temperature and pressure
fig, ax = plt.subplots()

hist_2d = np.histogram2d(
    datasets["jed0011"]["hc_top_temperature"].values,
    datasets["jed0011"]["hc_top_pressure"].values,
    bins=[np.linspace(180, 290, 60), np.linspace(0, 900, 60)],
    density=True,
)

pcol = ax.pcolormesh(
    hist_2d[2],
    hist_2d[1],
    np.log10(hist_2d[0]),
    cmap="viridis",
)

ax.axhline(273.15 - 35, color="grey", linestyle="--")
ax.axvline(350, color="grey", linestyle="--")
ax.set_ylabel("Temperature / K")
ax.set_xlabel("Pressure / hPa")
ax.spines[["top", "right"]].set_visible(False)
cb = fig.colorbar(pcol)
cb.set_label("log10(PDF)")
fig.savefig(f"plots/feedback/hc_top_2d.png", dpi=300, bbox_inches="tight")

# %% plot share of masked values
masks_temp = {}
binned_masks_temp = {}
masks_pressure = {}
binned_masks_pressure = {}
for run in runs:
    masks_temp[run] = datasets[run]["hc_top_temperature"] < (273.15 - 35)
    masks_pressure[run] = datasets[run]["hc_top_pressure"] < 350
    binned_masks_temp[run] = (
        masks_temp[run].groupby_bins(datasets[run]["iwp"], iwp_bins).mean()
    )
    binned_masks_pressure[run] = (
        masks_pressure[run].groupby_bins(datasets[run]["iwp"], iwp_bins).mean()
    )


fig, ax = plt.subplots()
for run in runs:
    binned_masks_temp[run].plot(ax=ax, label=exp_name[run], color=colors[run])
    binned_masks_pressure[run].plot(
        ax=ax, label=exp_name[run], linestyle="--", color=colors[run]
    )
ax.set_xscale("log")
ax.set_xlabel("$I$  / kg m$^{-2}$")
ax.set_ylabel("Share of unmasked values")
ax.spines[["top", "right"]].set_visible(False)
handles = [
    plt.Line2D([0], [0], color="grey", linestyle="-"),
    plt.Line2D([0], [0], color="grey", linestyle="--"),
]
labels = ["Temperature", "Pressure"]
fig.legend(handles, labels, bbox_to_anchor=[0.7, 0], frameon=True, ncols=2)
fig.savefig(f"plots/feedback/mask_share.png", dpi=300, bbox_inches="tight")

# %% plot high cloud temperature for multiple pressure thresholds
p_thresholds = [1000, 500, 350, 250, 200]
fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=True, sharey=True)
for i, p in enumerate(p_thresholds):
    ax = axes.flatten()[i]
    for run in runs:
        datasets[run]["hc_top_temperature"].where(
            datasets[run]["hc_top_pressure"] < p
        ).groupby_bins(datasets[run]["iwp"], iwp_bins).mean().plot(
            ax=ax, label=exp_name[run], color=colors[run]
        )
    ax.set_title(f"$p_{{\mathrm{{hc}}}} < {p}$ hPa")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim([1e-4, 4e1])
    ax.set_xscale("log")
    ax.set_xlabel("$I$  / kg m$^{-2}$")
    ax.set_ylabel(r"$T_{\mathrm{hc}}$  / K")

axes[2, 1].remove()
fig.tight_layout()
fig.savefig("plots/feedback/hc_top_pressure_thresholds.png", dpi=300)

# %% plot histograms of cloud top temperature
hists_p = {}
percentile_90 = {}
median = {}
fig, ax = plt.subplots()
for run in runs:
    percentile_90[run] = datasets[run]["hc_top_temperature"].quantile(0.90).values
    median[run] = datasets[run]["hc_top_temperature"].quantile(0.5).values
    hists_p[run] = np.histogram(
        datasets[run]["hc_top_temperature"].values,
        bins=np.linspace(180, 280, 70),
        density=True,
    )
    ax.stairs(hists_p[run][0], hists_p[run][1], label=exp_name[run], color=colors[run])
    ax.axvline(median[run], color=colors[run], linestyle="--", alpha=0.5)
    ax.axvline(273.15 - 35, color="grey")
    ax.axvline(percentile_90[run], color=colors[run], linestyle=":")
ax.set_xlabel("$T_{\mathrm{hc}}$  / K")
ax.set_ylabel("PDF")
ax.spines[["top", "right"]].set_visible(False)

labels = ["Median", "90th percentile", "-35°C (Haslehner et al., 2024)"]
handles = [
    plt.Line2D([0], [0], color="grey", linestyle="--"),
    plt.Line2D([0], [0], color="grey", linestyle=":"),
    plt.Line2D([0], [0], color="grey"),
]

ax.legend(handles, labels, frameon=True, ncols=3, bbox_to_anchor=[1, -0.12])
fig.savefig("plots/feedback/hc_top_temperature_hist.png", dpi=300, bbox_inches="tight")


# %% plot hc_top_temperature for profiles with hc_top_pressure < percentile_90
fig, ax = plt.subplots()

for run in runs:
    datasets[run]["hc_top_temperature"].where(
        datasets[run]["hc_top_temperature"] < percentile_90[run]
    ).groupby_bins(
        datasets[run]["iwp"],
        iwp_bins,
    ).mean().plot(
        ax=ax, label=exp_name[run], color=colors[run]
    )
ax.set_xscale("log")
ax.set_xlabel("$I$  / kg m$^{-2}$")
ax.set_ylabel(r"$T_{\mathrm{hc}}$  / K")
ax.spines[["top", "right"]].set_visible(False)

# %% check at which temperature there is no more liquid condensate
share_liquid = (datasets["jed0011"]["clw"] + datasets["jed0011"]["qr"]) / (
    datasets["jed0011"]["cli"]
    + datasets["jed0011"]["qs"]
    + datasets["jed0011"]["qg"]
    + datasets["jed0011"]["clw"]
    + datasets["jed0011"]["qr"]
)
# %%
fig, ax = plt.subplots()
t_bins = np.linspace(180, 300, 100)
t_points = (t_bins[1:] + t_bins[:-1]) / 2
binned_share = share_liquid.sel(index=slice(0, 1e3)).groupby_bins(datasets['jed0011']['ta'].sel(index=slice(0, 1e3)), t_bins).mean()

ax.scatter(
    datasets['jed0011']['ta'].sel(index=slice(0, 1e3)),
    share_liquid.sel(index=slice(0, 1e3)),
    s=0.1, 
    alpha=0.1
)
ax.plot(t_points, binned_share, color="k")
ax.axvline(273.15 - 35, color="grey", linestyle="--")
ax.set_xlabel("Temperature / K")
ax.set_ylabel("Share of liquid condensate")
ax.spines[["top", "right"]].set_visible(False)

# %%
