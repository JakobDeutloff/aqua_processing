# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from src.calc_variables import calc_IWC_cumsum, calculate_hc_temperature

# %% load data
runs = ["jed0011", "jed0022", "jed0033"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
colors = {"jed0011": "k", "jed0022": "r", "jed0033": "orange"}
datasets = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed.nc"
    ).sel(index=slice(None, 1e6))

vgrid = xr.open_dataset(
    "/work/bm1183/m301049/icon-mpim/experiments/jed0001/atm_vgrid_angel.nc"
).mean("ncells").rename({"height_2": "height", "height": "height_2"})

# %% calculate brightness temp from fluxes
def calc_brightness_temp(flux):
    return (flux / 5.67e-8) ** (1 / 4)

# %% interpolate datasets to the equally spaced height grid
ds_interp = {}
z_grid = np.linspace(vgrid['zg'].max().values, vgrid['zg'].min().values, 100)
height_grid = np.flip(np.interp(np.flip(z_grid), np.flip(vgrid['zg'].values), np.flip(datasets['jed0011']['height'].values)))
for run in runs:
    print(run)
    ds_interp[run] = datasets[run][['ta', 'cli', 'qs', 'qg', 'phalf']].interp(height=height_grid, method='quintic')
    ds_interp[run]['dzghalf'] = np.abs(np.diff(z_grid).mean())
    ds_interp[run] = ds_interp[run].assign(iwc_cumsum=calc_IWC_cumsum(ds_interp[run]))

# %% determine tropopause height and clearsky
masks_troposphere = {}
for run in runs:
    mask_stratosphere = z_grid < 25e3
    idx_trop = ds_interp[run]["ta"].where(mask_stratosphere).argmin("height")
    height_trop = ds_interp[run]["height"].isel(height=idx_trop)
    masks_troposphere[run] = (ds_interp[run]["height"] > height_trop)

#%% only look at clouds with cloud tops above 350 hPa and IWP > 1e-1 so that e = 1 can be assumed
ice_cumsums = {}
for run in runs:
    mask = (datasets[run]["iwp"] > 1e-1) 
    T_bright = calc_brightness_temp(datasets[run]['rlut'])
    height_bright_idx = np.abs(ds_interp[run]['ta'].where(masks_troposphere[run]) - T_bright).argmin("height")  # fills with 0 for NaNs
    ice_cumsums[run] = ds_interp[run]["iwc_cumsum"].isel(height=height_bright_idx)
    mean_ice_cumsum = ice_cumsums[run].where(mask).median()
    print(f"{run} {mean_ice_cumsum.values}")

# %% plot binned iwc_cumsum by iwp 
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
iwp_bins = np.logspace(-1, 1, 20)
for run in runs:
    ice_cumsums[run].groupby_bins(
        datasets[run]["iwp"],
        iwp_bins,
    ).mean().plot(
        ax=ax,
        color=colors[run],
    )

ax.set_xscale("log")

# %% plot inverted brightness temperature
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
for run in runs:
    T_bright = calc_brightness_temp(datasets[run]['rlut'].where(datasets[run]['iwp'] > 1e-1))
    T_bright.groupby_bins(
        datasets[run]["iwp"],
        iwp_bins,
    ).mean().plot(
        ax=ax,
        color=colors[run],
    )
ax.set_xscale("log")

# %% calculate hc temperature with flux inversion 
def calculate_hc_temperature_bright(ds, z_grid):
    """
    Calculate the temperature of high clouds.
    """
    
    # calculate mask troposphere 
    mask_stratosphere = z_grid < 25e3
    idx_trop = ds["ta"].where(mask_stratosphere).argmin("height")
    height_trop = ds["height"].isel(height=idx_trop)
    mask_troposphere = ds["height"] > height_trop

    # calculate brightness temperature
    T_bright = calc_brightness_temp(ds["rlut"])
    top_idx_thick = np.abs(ds["ta"].where(mask_troposphere) - T_bright).argmin("height")
    top_idx_thin = (ds["cli"] + ds["qs"] + ds["qg"]).argmax("height")

    top_idx = xr.where(top_idx_thick < top_idx_thin, top_idx_thick, top_idx_thin)
    p_top = ds.isel(height=top_idx).phalf
    T_h = ds["ta"].isel(height=top_idx)

    T_h.attrs = {"units": "K", "long_name": "High Cloud Top Temperature"}
    p_top.attrs = {"units": "hPa", "long_name": "High cloud top pressure"}

    return T_h, p_top / 100


# %% calculate hc temperature
hc_temp = {}
for run in runs:
    hc_temp[run], _ = calculate_hc_temperature_bright(datasets[run], vgrid['zg'])

# %%
iwp_bins = np.logspace(-4, 1, 50)
binned_temps = {}
binned_temps_bright = {}
for run in runs:
    binned_temps[run] = (
        datasets[run]["hc_top_temperature"]
        .groupby_bins(datasets[run]["iwp"], iwp_bins)
        .mean()
    )
    binned_temps_bright[run] = (
        hc_temp[run]
        .groupby_bins(datasets[run]["iwp"], iwp_bins)
        .mean()
    )
# %% plot hc temperatures 
fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

for run in runs:
    binned_temps[run].plot(
        ax=axes[0],
        color=colors[run],
        label=exp_name[run],
    )
    binned_temps_bright[run].plot(
        ax=axes[0],
        color=colors[run],
        linestyle='--',
    )

axes[1].axhline(0, color="grey", linestyle="--")
for run in runs[1:]:
    (binned_temps[run] - binned_temps[runs[0]]).plot(
        ax=axes[1],
        color=colors[run],
        label=exp_name[run],
    )
    (binned_temps_bright[run] - binned_temps_bright[runs[0]]).plot(
        ax=axes[1],
        color=colors[run],
        linestyle='--',
    )
axes[0].set_xscale("log")

# %%
