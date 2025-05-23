# %%
import xarray as xr
from src.calc_variables import calc_cre
import numpy as np
import matplotlib.pyplot as plt

# %% load data
runs = ["jed0011", "jed0022", "jed0033"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
datasets = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed.nc"
    )
# %% construct mask height to exclude the same number of values for every iwp bin
masks_height = {}
mask_type = "raw"
if mask_type == "dist_filter":
    iwp_bins = np.logspace(-4, np.log10(40), 51)
    masks_height["jed0011"] = datasets["jed0011"]["hc_top_temperature"] < datasets[
        "jed0011"
    ]["hc_top_temperature"].where(datasets["jed0011"]["iwp"] > 1e-4).quantile(0.90)
    quantiles = (
        (masks_height["jed0011"] * 1)
        .groupby_bins(datasets["jed0011"]["iwp"], iwp_bins)
        .mean()
    )

    for run in runs[1:]:
        mask = xr.DataArray(
            np.ones_like(datasets[run]["hc_top_temperature"]),
            dims=datasets[run]["hc_top_temperature"].dims,
            coords=datasets[run]["hc_top_temperature"].coords,
        )
        for i in range(len(iwp_bins) - 1):
            mask_ds = (datasets[run]["iwp"] > iwp_bins[i]) & (
                datasets[run]["iwp"] <= iwp_bins[i + 1]
            )
            temp_vals = datasets[run]["hc_top_temperature"].where(mask_ds)
            mask_temp = temp_vals > temp_vals.quantile(quantiles[i])
            # mask n_masked values with the highest temperatures from temp_vals
            mask = xr.where(mask_ds & mask_temp, 0, mask)
        masks_height[run] = mask
elif mask_type == "simple_filter":
    for run in runs:
        masks_height[run] = datasets[run]["hc_top_temperature"] < (273.15 - 35)
elif mask_type == "raw":
    for run in runs:
        masks_height[run] = True

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
# %% bin CRE
iwp_bins = np.logspace(-4, np.log10(40), 51)
iwp_points = (iwp_bins[:-1] + iwp_bins[1:]) / 2
cre_arr = {
    "net": ("iwp", np.zeros_like(iwp_points)),
    "sw": ("iwp", np.zeros_like(iwp_points)),
    "lw": ("iwp", np.zeros_like(iwp_points)),
}

cre_interp = {
    "jed0011": cre_arr.copy(),
    "jed0022": cre_arr.copy(),
    "jed0033": cre_arr.copy(),
}

for run in runs:
    cre_interp[run]["net"] = (
        datasets[run]["cre_net_hc"]
        .where(masks_height[run])
        .groupby_bins(datasets[run]["iwp"], iwp_bins)
        .mean()
    )
    cre_interp[run]["sw"] = (
        datasets[run]["cre_sw_hc"]
        .where(masks_height[run])
        .groupby_bins(datasets[run]["iwp"], iwp_bins)
        .mean()
    )
    cre_interp[run]["lw"] = (
        datasets[run]["cre_lw_hc"]
        .where(masks_height[run])
        .groupby_bins(datasets[run]["iwp"], iwp_bins)
        .mean()
    )

# %% convert to xarray
for run in runs:
    cre_interp[run] = xr.Dataset(
        {
            "net": (["iwp"], cre_interp[run]["net"].values),
            "sw": (["iwp"], cre_interp[run]["sw"].values),
            "lw": (["iwp"], cre_interp[run]["lw"].values),
        },
        coords={
            "iwp": ("iwp", iwp_points),
        },
    )
    cre_interp[run].attrs = {
        "units": "W m^-2",
        "long_name": "Cloud Radiative Effect",
    }

# %% plot difference between CREs
fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 10),
    sharex=True,
)
colors = {
    "jed0011": "black",
    "jed0022": "red",
    "jed0033": "orange",
}
for run in runs[1:]:
    axes[0].plot(
        cre_interp[run]["net"]["iwp"].sel(iwp=slice(1e-4, 10)),
        (cre_interp[run]["net"] - cre_interp[runs[0]]["net"]).sel(iwp=slice(1e-4, 10)),
        color=colors[run],
        label=exp_name[run],
    )
    axes[1].plot(
        cre_interp[run]["sw"]["iwp"].sel(iwp=slice(1e-4, 10)),
        (cre_interp[run]["sw"] - cre_interp[runs[0]]["sw"]).sel(iwp=slice(1e-4, 10)),
        color=colors[run],
        label=exp_name[run],
    )
    axes[2].plot(
        cre_interp[run]["lw"]["iwp"].sel(iwp=slice(1e-4, 10)),
        (cre_interp[run]["lw"] - cre_interp[runs[0]]["lw"]).sel(iwp=slice(1e-4, 10)),
        label=exp_name[run],
        color=colors[run],
    )

for ax in axes:
    ax.set_xscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="grey")

axes[-1].set_xlabel("IWP / kg m$^{-2}$")
axes[0].set_ylabel("CRE net / W m$^{-2}$")
axes[1].set_ylabel("CRE sw / W m$^{-2}$")
axes[2].set_ylabel("CRE lw / W m$^{-2}$")

# %% save cre
for run in runs:
    cre_interp[run].to_netcdf(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/cre/{run}_cre_{mask_type}.nc"
    )
# %%
