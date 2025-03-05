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
    datasets[run] = datasets[run].assign(time_local=lambda d: d.time.dt.hour + (d.clon / 15))
    datasets[run]["time_local"] = datasets[run]["time_local"].where(
        datasets[run]["time_local"] < 24, datasets[run]["time_local"] - 24
    ).where(
        datasets[run]["time_local"] > 0, datasets[run]["time_local"] + 24
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


# %% interpolate and average cre
cre_arr = {}
cre_interp = {}
cre_interp_mean = {}
cre_interp_std = {}
iwp_bins = np.logspace(-4, np.log10(40), 51)
time_bins = np.linspace(0, 24, 25)
for run in runs:
    cre_arr[run], cre_interp[run], cre_interp_mean[run] = bin_and_average_cre(
        datasets[run], iwp_bins, time_bins, mask_height=datasets[run]["hc_top_pressure"] < (350)
    )

# %% save processed data
for run in runs:
    path = f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/"
    file = path + f"cre/{run}_cre_arr_rand_p.nc"
    if os.path.exists(file):
        os.remove(file)
    cre_arr[run].to_netcdf(file)
    file = path + f"cre/{run}_cre_interp_rand_p.nc"
    if os.path.exists(file):
        os.remove(file)
    cre_interp[run].to_netcdf(file)
    file = path + f"cre/{run}_cre_interp_mean_rand_p.nc"
    if os.path.exists(file):
        os.remove(file)
    cre_interp_mean[run].to_netcdf(file)


# %%
