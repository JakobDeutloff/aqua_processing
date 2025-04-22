# %%
from scipy.optimize import least_squares
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from src.calc_variables import calculate_hc_temperature, calc_IWC_cumsum

# %%
runs = ["jed0011", "jed0022", "jed0033"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
colors = {'jed0011': 'k', 'jed0022': 'r', 'jed0033': 'orange'}
datasets = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed_20_conn.nc"
    ).sel(index=slice(None, 1e6))


# %% calculate hc top temperature
run = 'jed0022'
hc_top_temp, hc_top_pressure = calculate_hc_temperature(datasets[run], 0.06)

# %% mask for parameterization
mask = (hc_top_temp < 273 - 35) & ~datasets[run]["mask_low_cloud"]

# %% calculate high cloud emissivity
sigma = 5.67e-8  # W m-2 K-4
LW_out_as = datasets[run]["rlut"]
LW_out_cs = datasets[run]["rlutcs"]
rad_hc = hc_top_temp**4 * sigma
hc_emissivity = (LW_out_as - LW_out_cs) / (rad_hc - LW_out_cs)
hc_emissivity = xr.where(
    (hc_emissivity < -0.1) | (hc_emissivity > 1.5), np.nan, hc_emissivity
)

# %% aveage over IWP bins
IWP_bins = np.logspace(-4, 1, num=50)
IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2
mean_hc_emissivity = (
    hc_emissivity.where(mask)
    .groupby_bins(
        datasets[run]["iwp"],
        IWP_bins,
        labels=IWP_points,
    )
    .mean()
)
mean_hc_temp = (
    hc_top_temp.where(mask)
    .groupby_bins(
        datasets[run]["iwp"],
        IWP_bins,
        labels=IWP_points,
    )
    .mean()
)


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
    datasets[run]["iwp"].where(mask).sel(index=slice(0, 1e5)),
    hc_emissivity.where(mask).sel(index=slice(0, 1e5)),
    s=1,
    color="k",
    alpha=0.5,
)

axes[0].plot(IWP_points, logistic_curve, color="r", label="logistic fit")
axes[0].axhline(1, color="green", linestyle="--")

axes[1].plot(IWP_points, mean_hc_temp)
axes[1].scatter(
    datasets[run]["iwp"].where(mask).sel(index=slice(0, 1e5)),
    hc_top_temp.where(mask).sel(index=slice(0, 1e5)),
    s=1,
    color="k",
    alpha=0.5,
)

for ax in axes:
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1e1)
# %%
