# %%
from scipy.optimize import least_squares
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
import sys

# %%
# Set CWD to PYTHONPATH
pythonpath = os.environ.get("PYTHONPATH", "")
if pythonpath:
    os.chdir(pythonpath)
    print(f"Working directory set to: {os.getcwd()}")
else:
    print("PYTHONPATH is not set.")

# %%
run = sys.argv[1]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
colors = {'jed0011': 'k', 'jed0022': 'r', 'jed0033': 'orange'}
ds = xr.open_dataset(
    f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed.nc"
).sel(index=slice(None, 1e6))

# %% initialize datasets
sw_vars = xr.Dataset()
mean_sw_vars = pd.DataFrame()

# %% set mask
mask_parameterisation = (ds['mask_low_cloud'] == 0) & (ds['hc_top_temperature'] < 255) 

# %% calculate high cloud albedo
def calc_hc_albedo(a_cs, a_as):
    return (a_as - a_cs) / (a_cs * (a_as - 2) + 1)


sw_vars["wetsky_albedo"] = np.abs(ds["rsutws"] / ds["rsdt"])
sw_vars["allsky_albedo"] = np.abs(ds["rsut"] / ds["rsdt"])
sw_vars["clearsky_albedo"] = np.abs(ds["rsutcs"] / ds["rsdt"])
cs_albedo = xr.where(
    ds["conn"], sw_vars["clearsky_albedo"], sw_vars["wetsky_albedo"]
)
sw_vars["high_cloud_albedo"] = calc_hc_albedo(cs_albedo, sw_vars["allsky_albedo"])

# %% calculate mean albedos by weighting with the incoming SW radiation in IWP bins
IWP_bins = np.logspace(-4, 1, num=50)
IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2
time_bins = np.linspace(0, 24, 25)
time_points = (time_bins[1:] + time_bins[:-1]) / 2
binned_hc_albedo = np.zeros([len(IWP_bins) - 1, len(time_bins) - 1]) * np.nan

for i in range(len(IWP_bins) - 1):
    IWP_mask = (ds["iwp"] > IWP_bins[i]) & (ds["iwp"] <= IWP_bins[i + 1])
    for j in range(len(time_bins) - 1):
        time_mask = (ds['time_local'] > time_bins[j]) & (
            ds['time_local'] <= time_bins[j + 1]
        )
        binned_hc_albedo[i, j] = float(
            sw_vars["high_cloud_albedo"]
            .where(IWP_mask & mask_parameterisation & time_mask)
            .mean()
            .values
        )

# %% plot albedo in SW bins
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

pcol = ax.pcolormesh(IWP_bins, time_bins, binned_hc_albedo.T, cmap="viridis")

ax.set_ylabel("Local time")

ax.set_xscale("log")
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_xlim([1e-4, 1e1])

fig.colorbar(pcol, label="High Cloud Albedo", location="bottom", ax=ax, shrink=0.8)

# %% average over SW albedo bins
mean_hc_albedo = np.zeros(len(IWP_points))
mean_hc_albed_interp = np.zeros(len(IWP_points))
SW_weights = np.zeros(len(time_points))
for i in range(len(time_points)):
    time_mask = (ds['time_local'] > time_bins[i]) & (
        ds['time_local'] <= time_bins[i + 1]
    )
    SW_weights[i] = float(
        ds['rsdt']
        .where(time_mask)
        .mean()
        .values
    )

for i in range(len(IWP_bins) - 1):
    nan_mask = ~np.isnan(binned_hc_albedo[i, :])
    mean_hc_albedo[i] = np.sum(
        binned_hc_albedo[i, :][nan_mask] * SW_weights[nan_mask]
    ) / np.sum(SW_weights[nan_mask])
    nan_mask_interp = ~np.isnan(binned_hc_albedo[i, :])
    mean_hc_albed_interp[i] = np.sum(
        binned_hc_albedo[i, :][nan_mask] * SW_weights[nan_mask]
    ) / np.sum(SW_weights[nan_mask])


mean_sw_vars.index = IWP_points
mean_sw_vars.index.name = "IWP"
mean_sw_vars["binned_albedo"] = mean_hc_albedo
mean_sw_vars["interpolated_albedo"] = mean_hc_albed_interp

# %% fit logistic function to mean albedo
# prepare x and required y data
x = np.log10(IWP_points)
y = mean_sw_vars["interpolated_albedo"].copy()
nan_mask = ~np.isnan(y)
x = x[nan_mask]
y = y[nan_mask]

# initial guess
p0 = [0.75, -1.3, 1.9]


def logistic(params, x):
    return params[0] / (1 + np.exp(-params[2] * (x - params[1])))


def loss(params):
    return logistic(params, x) - y


res = least_squares(loss, p0, xtol=1e-12)
logistic_curve = logistic(res.x, np.log10(IWP_points))

# %% plot fitted albedo in scatterplot with IWP


fig, ax = plt.subplots()

ax.scatter(ds['iwp'].where(mask_parameterisation).sel(index=slice(0, 1e5)),
           sw_vars["high_cloud_albedo"].where(mask_parameterisation).sel(index=slice(0, 1e5)),
           s=1,
           c=ds['rsdt'].where(mask_parameterisation).sel(index=slice(0, 1e5)),
           cmap="viridis",
           alpha=0.5,
           label="All Data"
)
ax.plot(mean_sw_vars["interpolated_albedo"], label="Mean Albedo", color="k")
ax.plot(
    IWP_points, logistic_curve, label="Fitted Logistic", color="red", linestyle="--"
)
ax.legend()
ax.set_xscale('log')
ax.set_xlim(1e-4, 1e1)

plt.show()

# %% save coefficients as pkl file
sw_vars.to_netcdf(f"data/{run}_sw_vars.nc")
with open(f"data/params/{run}_hc_albedo_params.pkl", "wb") as f:
    pickle.dump(res.x, f)
with open(f"data/{run}_sw_vars_mean.pkl", "wb") as f:
    pickle.dump(mean_sw_vars, f)

