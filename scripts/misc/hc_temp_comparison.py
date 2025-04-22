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


# %%
def calc_brightness_temp(flux):
    return (flux / 5.67e-8) ** (1 / 4)


def calculate_hc_temperature_bright(ds):
    """
    Calculate the temperature of high clouds.
    """
    T_bright = calc_brightness_temp(ds["rlut"])
    top_idx_thick = np.abs(ds["ta"] - T_bright).argmin("height")
    top_idx_thin = (ds["cli"] + ds["qs"] + ds["qg"]).argmax("height")

    top_idx = xr.where(top_idx_thick < top_idx_thin, top_idx_thick, top_idx_thin)
    p_top = ds.isel(height=top_idx).phalf
    T_h = ds["ta"].isel(height=top_idx)

    T_h.attrs = {"units": "K", "long_name": "High Cloud Top Temperature"}
    p_top.attrs = {"units": "hPa", "long_name": "High cloud top pressure"}

    return T_h, p_top / 100


# %%
T_hc_bright = {}
T_hc_bright_bin = {}
T_hc_mass = {}
T_hc_mass_bin = {}
IWP_bins = np.logspace(-4, 1, num=50)
IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2
for run in runs:
    T_hc_bright[run], _ = calculate_hc_temperature_bright(datasets[run])
    T_hc_mass[run], _ = calculate_hc_temperature(datasets[run], 0.089)
    T_hc_bright_bin[run] = (
        T_hc_bright[run]
        .where(T_hc_bright[run]<238)
        .groupby_bins(datasets[run]["iwp"], IWP_bins, labels=IWP_points)
        .mean()
    )
    T_hc_mass_bin[run] = (
        T_hc_mass[run]
        .where(T_hc_mass[run]<238)
        .groupby_bins(datasets[run]["iwp"], IWP_bins, labels=IWP_points)
        .mean()
    )

# %% plot temperatures
fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharey=True, sharex=True) 
for run in runs:
    axes[0].plot(
        IWP_points,
        T_hc_bright_bin[run],
        label=exp_name[run],
        color=colors[run],
    )
    axes[1].plot(
        IWP_points,
        T_hc_mass_bin[run],
        label=exp_name[run],
        color=colors[run],
    )

axes[2].plot(
    IWP_points,
    T_hc_mass_bin["jed0011"],
    color='k',
    linestyle='-'
)
axes[2].plot(
    IWP_points,
    T_hc_bright_bin["jed0011"],
    color='k',
    linestyle='--'
)
for ax in axes:
    ax.set_xscale("log")
# %% calculate share of excluded values 
for run in runs:
    print(f"{run}: {np.sum(T_hc_bright[run] > 273 - 35) / T_hc_bright[run].size:.2%} of values excluded by bright")
    print(f"{run}: {np.sum(T_hc_mass[run] > 273 - 35) / T_hc_mass[run].size:.2%} of values excluded by mass")