"""
Determine albedo and LW emissions of lower troposphere and derive respective conceptual model parameters
"""

# %% import
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.interpolate import griddata
import pandas as pd
import pickle
from matplotlib.colors import LinearSegmentedColormap, LogNorm
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

# %% read data
run = sys.argv[1]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
ds = xr.open_dataset(
    f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed.nc"
).sel(index=slice(None, 1e6))

# %% initialize dataset
lower_trop_vars = xr.Dataset()
binned_lower_trop_vars = pd.DataFrame()

# %% set masks
mask_low_cloud = ds["lwp"] > 1e-4
mask_connected = ds["conn"] == 1

# %% calculate mean lc fraction
iwp_bins = np.logspace(-4, 1, num=50)
f_mean = ds["mask_low_cloud"].where(ds['iwp']>1e-4).mean().values.round(3)

# %% calculate albedos
albedo_allsky = np.abs(ds["rsutws"] / ds["rsdt"])
albedo_clearsky = np.abs(ds["rsutcs"] / ds["rsdt"])
lower_trop_vars["albedo_allsky"] = albedo_allsky
lower_trop_vars["albedo_clearsky"] = albedo_clearsky
alpha_t = xr.where(ds["mask_low_cloud"], albedo_allsky, albedo_clearsky)
lower_trop_vars["alpha_t"] = alpha_t

# %% average and interpolate albedo
LWP_bins = np.logspace(-14, 2, num=150)
LWP_points = (LWP_bins[1:] + LWP_bins[:-1]) / 2
time_bins = np.linspace(0, 24, 25)
time_points = (time_bins[1:] + time_bins[:-1]) / 2
binned_lt_albedo = np.zeros([len(LWP_bins) - 1, len(time_bins) - 1]) * np.nan

for i in range(len(LWP_bins) - 1):
    LWP_mask = (ds["lwp"] > LWP_bins[i]) & (ds["lwp"] <= LWP_bins[i + 1])
    for j in range(len(time_bins) - 1):
        time_mask = (ds["time_local"] > time_bins[j]) & (
            ds["time_local"] <= time_bins[j + 1]
        )
        binned_lt_albedo[i, j] = float(
            lower_trop_vars["albedo_allsky"].where(LWP_mask & time_mask).mean().values
        )

# %% interpolate albedo bins
non_nan_indices = np.array(np.where(~np.isnan(binned_lt_albedo)))
non_nan_values = binned_lt_albedo[~np.isnan(binned_lt_albedo)]
nan_indices = np.array(np.where(np.isnan(binned_lt_albedo)))
interpolated_values = griddata(
    non_nan_indices.T, non_nan_values, nan_indices.T, method="linear"
)
binned_lt_albedo_interp = binned_lt_albedo.copy()
binned_lt_albedo_interp[np.isnan(binned_lt_albedo)] = interpolated_values

# %% plot albedo in SW bins
fig, axes = plt.subplots(1, 2, figsize=(10, 6))

pcol = axes[0].pcolormesh(LWP_bins, time_bins, binned_lt_albedo.T, cmap="viridis")
axes[1].pcolormesh(LWP_bins, time_bins, binned_lt_albedo_interp.T, cmap="viridis")

axes[0].set_ylabel("SWin at TOA / W m$^{-2}$")
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("LWP / kg m$^{-2}$")
    ax.set_xlim([1e-4, 5e1])

fig.colorbar(pcol, label="Allsky Albedo", location="bottom", ax=axes[:], shrink=0.8)

# %% average over SW  bins
SW_weights = np.zeros(len(time_points))
for i in range(len(time_points)):
    SW_weights[i] = float(
        ds["rsdt"]
        .where(
            (ds["time_local"] > time_bins[i]) & (ds["time_local"] <= time_bins[i + 1])
        )
        .mean()
        .values
    )

mean_lt_albedo = np.zeros(len(LWP_points))
mean_lt_albedo_interp = np.zeros(len(LWP_points))
for i in range(len(LWP_bins) - 1):
    nan_mask = ~np.isnan(binned_lt_albedo[i, :])
    mean_lt_albedo[i] = np.sum(
        binned_lt_albedo[i, :][nan_mask] * SW_weights[nan_mask]
    ) / np.sum(SW_weights[nan_mask])
    nan_mask_interp = ~np.isnan(binned_lt_albedo_interp[i, :])
    mean_lt_albedo_interp[i] = np.sum(
        binned_lt_albedo_interp[i, :][nan_mask] * SW_weights[nan_mask]
    ) / np.sum(SW_weights[nan_mask])
binned_albedos = pd.DataFrame(
    np.array([mean_lt_albedo_interp, mean_lt_albedo]).T,
    index=LWP_points,
    columns=["interpolated", "raw"],
)


#  %% calculate mean albedos
mask_lwp_bins = LWP_points > 1e-4
number_of_points = np.zeros(len(LWP_points))
for i in range(len(LWP_bins) - 1):
    LWP_mask = (ds["lwp"] > LWP_bins[i]) & (ds["lwp"] <= LWP_bins[i + 1])
    number_of_points[i] = (
        lower_trop_vars["albedo_allsky"]
        .where(LWP_mask & ~mask_connected & ds["mask_height"])
        .count()
        .values
    )

mean_cloud_albedo = float(
    np.sum(
        binned_albedos["interpolated"][mask_lwp_bins] * number_of_points[mask_lwp_bins]
    )
    / np.sum(number_of_points[mask_lwp_bins])
)

mean_clearsky_albedo = float(
    (
        (
            lower_trop_vars["albedo_clearsky"].where(
                ~ds["mask_low_cloud"] & ds["mask_height"],
            )
        )
        .mean()
        .values
    )
)

# %% plot albedo vs LWP
fig, ax = plt.subplots()
ax.scatter(
    ds["lwp"].where(ds["mask_height"]).sel(index=slice(0, 1e5)),
    lower_trop_vars["albedo_allsky"].where(ds["mask_height"]).sel(index=slice(0, 1e5)),
    c=ds["rsdt"].where(ds["mask_height"]).sel(index=slice(0, 1e5)),
    marker=".",
    s=1,
    cmap="viridis",
)

ax.set_xlim([1e-14, 1e2])
ax.set_ylim([0, 1])
ax.set_xscale("log")
ax.set_xlabel("LWP / kg m$^{-2}$")

ax.axhline(mean_clearsky_albedo, color="k", linestyle="--", label="Mean clearsky")
ax.axhline(mean_cloud_albedo, color="grey", linestyle="--", label="Mean low cloud")
ax.plot(binned_albedos["interpolated"], color="k", linestyle="-", label="Mean")
ax.legend()
fig.tight_layout()


# %% calculate R_t
lower_trop_vars["R_t"] = -1 * xr.where(
    (mask_connected | ~mask_low_cloud),
    ds['rlutcs'],
    ds['rlutws'],
)

# %% calculate mean R_t
mean_R_l = float(
    lower_trop_vars["R_t"].where(ds["mask_height"] & ds["mask_low_cloud"])
    .mean()
    .values
)
mean_R_cs = float(
    lower_trop_vars["R_t"].where(ds["mask_height"] & ~ds["mask_low_cloud"])
    .mean()
    .values
)

# %% linear regression of R_t vs IWP
x_data = np.log10(ds["iwp"].where(ds["mask_height"])).values.flatten()
y_data = lower_trop_vars["R_t"].where(ds["mask_height"]).values.flatten()
nan_mask = ~np.isnan(x_data) & ~np.isnan(y_data)
x_data = x_data[nan_mask]
y_data = y_data[nan_mask]
y_data = y_data - np.mean(y_data)
c_h20_coeffs = linregress(x_data, y_data)

# %% plot R_t vs IWP and regression
fig, ax = plt.subplots()

colors = ["black", "grey", "blue"]
cmap = LinearSegmentedColormap.from_list("my_cmap", colors)
sc_rt = ax.scatter(
    ds["iwp"].where(ds["mask_height"]),
    lower_trop_vars["R_t"].where(
        ds["mask_height"]),
    s=0.1,
    c=ds['lwp'].where(ds["mask_height"]),
    cmap=cmap,
    norm=LogNorm(vmin=1e-4, vmax=1e2),
)
iwp_bins = np.logspace(-4, 1, num=50)
iwp_points = (iwp_bins[1:] + iwp_bins[:-1]) / 2
binned_r_t = (
    lower_trop_vars["R_t"].where(ds["mask_height"])
    .groupby_bins(ds["iwp"].where(ds["mask_height"]), iwp_bins)
    .mean()
)
ax.plot(iwp_points, binned_r_t, color="orange", label="Mean")
ax.axhline(mean_R_cs, color="grey", linestyle="--", label="Clearsky")
ax.axhline(mean_R_l, color="navy", linestyle="--", label="Low Cloud")
fit = (
    f_mean * mean_R_l
    + (1 - f_mean) * mean_R_cs
    + c_h20_coeffs.slope * np.log10(iwp_points)
    + c_h20_coeffs.intercept
)
ax.plot(iwp_points, fit, color="red", label="Fit")
ax.set_xlim([1e-5, 1e1])

ax.set_ylabel(r"LT LW Emissions ($\mathrm{R_t}$) / $\mathrm{W ~ m^{-2}}$")
ax.legend()
ax.set_xscale("log")

# %% calculate lower tropospheric variables binned by iwp
binned_lower_trop_vars["a_t"] = (
    lower_trop_vars["alpha_t"]
    .groupby_bins(ds["iwp"], iwp_bins)
    .mean()
)
binned_lower_trop_vars["R_t"] = (
    lower_trop_vars["R_t"]
    .groupby_bins(ds['iwp'], iwp_bins)
    .mean()
)

# %% save variables

lower_trop_vars.to_netcdf(f"data/{run}_lower_trop_vars.nc")
with open(f"data/{run}_lower_trop_vars_mean.pkl", "wb") as f:
    pickle.dump(binned_lower_trop_vars, f)
with open(f"data/params/{run}_C_h2o_params.pkl", "wb") as f:
    pickle.dump(c_h20_coeffs, f)
with open(f"data/params/{run}_lower_trop_params.pkl", "wb") as f:
    pickle.dump(
        {
            "a_l": mean_cloud_albedo,
            "a_cs": mean_clearsky_albedo,
            "R_l": mean_R_l,
            "R_cs": mean_R_cs,
            "f": f_mean,
        },
        f,
    )

# %%
