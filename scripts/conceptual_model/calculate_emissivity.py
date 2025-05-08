# %%
from scipy.optimize import least_squares
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import sys
import os

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

# %% initialize dataset for new variables
lw_vars = xr.Dataset()
mean_lw_vars = pd.DataFrame()


# %% mask for parameterization
mask = (ds['mask_low_cloud'] == 0) 
# %% calculate high cloud emissivity
sigma = 5.67e-8  # W m-2 K-4
LW_out_as = ds["rlut"]
LW_out_cs = ds["rlutcs"]
rad_hc = ds['hc_top_temperature']**4 * sigma
hc_emissivity = (LW_out_as - LW_out_cs) / (rad_hc - LW_out_cs)
hc_emissivity = xr.where(
    (hc_emissivity < -0.1) | (hc_emissivity > 1.5), np.nan, hc_emissivity
)
lw_vars["high_cloud_emissivity"] = hc_emissivity

# %% aveage over IWP bins
IWP_bins = np.logspace(-4, 1, num=50)
IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2
mean_hc_emissivity = (
    hc_emissivity.where(mask)
    .groupby_bins(
        ds["iwp"],
        IWP_bins,
        labels=IWP_points,
    )
    .mean()
)
mean_hc_temp = (
    ds['hc_top_temperature'].where(mask)
    .groupby_bins(
        ds["iwp"],
        IWP_bins,
        labels=IWP_points,
    )
    .mean()
)
mean_lw_vars.index = IWP_points
mean_lw_vars.index.name = "IWP"
mean_lw_vars["binned_emissivity"] = mean_hc_emissivity


# %% fit logistic function to mean high cloud emissivity

# prepare x and required y data
x = np.log10(IWP_points)
y = mean_hc_emissivity.copy()
nan_mask = ~np.isnan(y)
y = y[nan_mask]
x = x[nan_mask]


#initial guess
p0 = [ -2, 3, 1]

def logistic(params, x):
    return params[2] / (1 + np.exp(-params[1] * (x - params[0])))


def loss(params):
    return (logistic(params, x) - y) 

res = least_squares(loss, p0)
logistic_curve = logistic(res.x, np.log10(IWP_points))

# %% plot
fig, axes = plt.subplots(2, 1, figsize=(8, 8))
axes[0].plot(IWP_points, mean_hc_emissivity)
axes[0].scatter(
    ds["iwp"].where(mask).sel(index=slice(0, 1e5)),
    hc_emissivity.where(mask).sel(index=slice(0, 1e5)),
    s=1,
    color="k",
    alpha=0.5,
)

axes[0].plot(IWP_points, logistic_curve, color="r", label="logistic fit")
axes[0].axhline(1, color="green", linestyle="--")

axes[1].plot(IWP_points, mean_hc_temp)
axes[1].scatter(
    ds["iwp"].where(mask).sel(index=slice(0, 1e5)),
    ds['hc_top_temperature'].where(mask).sel(index=slice(0, 1e5)),
    s=1,
    color="k",
    alpha=0.5,
)

for ax in axes:
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1e1)

# %% save coefficients as pkl file
lw_vars.to_netcdf(f"data/{run}_lw_vars.nc")
with open(f"data/params/{run}_hc_emissivity_params.pkl", "wb") as f:
    pickle.dump(np.array([1., res.x[0], res.x[1]]), f)
with open(f"data/{run}_lw_vars_mean.pkl", "wb") as f:
    pickle.dump(mean_lw_vars, f)

# %%
