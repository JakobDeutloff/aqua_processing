# %%
import xarray as xr
from src.calc_variables import calc_cre, bin_and_average_cre
import numpy as np
import os

# %% load data
runs = ["jed0011", "jed0022", "jed0033"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
datasets = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed.nc"
    ).sel(index=slice(0, 5e6))
# %% calculate masks 
mask_type = "dist_filter"  
masks_height = {}
if mask_type == "raw":
    for run in runs:
        masks_height[run] = True   
elif mask_type == 'dist_filter':
    masks_height = {}
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


# %% interpolate and average cre
cre_arr = {}
cre_interp = {}
cre_interp_mean = {}
cre_interp_std = {}
iwp_bins = np.logspace(-4, np.log10(40), 51)
time_bins = np.linspace(0, 24, 25)
for run in runs:
    cre_arr[run], cre_interp[run], cre_interp_mean[run] = bin_and_average_cre(
        datasets[run], iwp_bins, time_bins, mask_height=masks_height[run]
    )

# %% save processed data
for run in runs:
    path = f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/cre/"
    file = path + f"{run}_cre_interp_{mask_type}.nc"
    if os.path.exists(file):
        os.remove(file)
    cre_interp_mean[run].to_netcdf(file)


# %%
