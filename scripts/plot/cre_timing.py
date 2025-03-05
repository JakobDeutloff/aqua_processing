# %%
import xarray as xr
import matplotlib.pyplot as plt
from src.calc_variables import calc_cre
import numpy as np

# %% load CRE data
runs = ["jed0011", "jed0022", "jed0033"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
cre_interp_mean = {}
cre_arr = {}
cre_arr_interp = {}
datasets = {}
for run in runs:
    cre_interp_mean[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/cre/{run}_cre_interp_mean_rand_t.nc"
    )
    cre_arr[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/cre/{run}_cre_arr_rand_t.nc"
    )
    cre_arr_interp[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/cre/{run}_cre_interp_rand_t.nc"
    )
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed.nc"
    )


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
iwp_bins = np.logspace(-4, np.log10(40), 51)
cre_net_hc_binned = {}
cre_sw_hc_binned = {}
cre_lw_hc_binned = {}
time_binned = {}
time_std = {}
for run in runs:
    cre_net_hc_binned[run] = (
        datasets[run]["cre_net_hc"]
        .where(datasets[run]["hc_top_temperature"] < 238.15)
        .groupby_bins(datasets[run]["iwp"], iwp_bins)
        .mean()
    )
    cre_sw_hc_binned[run] = (
        datasets[run]["cre_sw_hc"]
        .where(datasets[run]["hc_top_temperature"] < 238.15)
        .groupby_bins(datasets[run]["iwp"], iwp_bins)
        .mean()
    )
    cre_lw_hc_binned[run] = (
        datasets[run]["cre_lw_hc"]
        .where(datasets[run]["hc_top_temperature"] < 238.15)
        .groupby_bins(datasets[run]["iwp"], iwp_bins)
        .mean()
    )
    time_binned[run] = (
        datasets[run]["time_local"]
        .where(datasets[run]["hc_top_temperature"] < 238.15)
        .groupby_bins(datasets[run]["iwp"], iwp_bins)
        .mean()
    )
    time_std[run] = (
        datasets[run]["time_local"]
        .where(datasets[run]["hc_top_temperature"] < 238.15)
        .groupby_bins(datasets[run]["iwp"], iwp_bins)
        .std()
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

# %% plot mean time of iwp
fig, ax = plt.subplots()
colors = {"jed0011": "k", "jed0022": "red", "jed0033": "orange"}
for run in runs:
    time_binned[run].plot(ax=ax, label=exp_name[run], color=colors[run])

ax.set_xscale("log")
ax.set_xlabel("$I$ / $kg m^{-2}$")
ax.set_ylabel("Local Time / h")
ax.set_xlim([1e-4, 40])
ax.spines[["top", "right"]].set_visible(False)
ax.legend()
fig.savefig("plots/cre/mean_time_of_iwp.png")


# %%

# %%
