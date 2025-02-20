# %%
import xarray as xr
from src.calc_variables import calc_cre, bin_and_average_cre
import numpy as np
import os

# %% load data
runs = ["jed0011"]
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

# %% calculate cre high clouds
for run in runs:
    datasets[run] = datasets[run].assign(
        cre_net_hc=xr.where(
            datasets[run]['mask_low_cloud'],
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
            datasets[run]['mask_low_cloud'],
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
            datasets[run]['mask_low_cloud'],
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
iwp_bins = np.logspace(-5, np.log10(40), 51)
time_bins = np.linspace(0, 24, 25)
for run in ["jed0011"]:
    cre_arr[run], cre_interp[run], cre_interp_mean[run] = bin_and_average_cre(
        datasets[run], iwp_bins, time_bins
    )

# %% plot cre
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4))
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
cre_interp_mean["jed0011"]["net"].plot(ax=ax, color='k')
cre_interp_mean["jed0011"]["sw"].plot(ax=ax, color="blue")
cre_interp_mean["jed0011"]["lw"].plot(ax=ax, color="red")

ax.set_xlim([1e-5, 2e1])
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('$I$  / kg m$^{-2}$')
ax.set_ylabel('$C$  / W m$^{-2}$')
ax.set_xscale("log")
fig.savefig('plots/cre_iwp.png', dpi=300)

# %% check differences in flux
datasets["jed0011"]["rsutcs"].where(
    (datasets["jed0011"]["clivi"] < 1e-8)
    & (datasets["jed0011"]["cllvi"] < 1e-8)
    & (datasets["jed0011"]["qrvi"] < 1e-8)
    & (datasets["jed0011"]["qsvi"] < 1e-8)
    & (datasets["jed0011"]["qgvi"] < 1e-8)
).mean()

# %% save processed data
for run in runs:
    path = f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/"
    if os.path.exists(path + f"random_sample/{run}_randsample_processed.nc"):
        os.remove(path + f"random_sample/{run}_randsample_processed.nc")
    datasets[run].to_netcdf(path + f"random_sample/{run}_randsample_processed.nc")
    cre_arr[run].to_netcdf(path + f"cre/{run}_cre_arr_rand.nc")
    cre_interp[run].to_netcdf(path + f"cre/{run}_cre_interp_rand.nc")
    cre_interp_mean[run].to_netcdf(path + f"cre/{run}_cre_interp_mean_rand.nc")
    cre_interp_std[run].to_netcdf(path + f"cre/{run}_cre_interp_std_rand.nc")


# %% investigate connectedness 
fig, ax = plt.subplots()
datasets['jed0011']['conn'].groupby_bins(datasets['jed0011']['iwp'], np.logspace(-5, np.log10(40), 51)).mean().plot(ax=ax)
ax.set_xscale('log')


# %%
