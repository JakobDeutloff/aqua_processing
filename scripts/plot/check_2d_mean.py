# %% import
import xarray as xr
import matplotlib.pyplot as plt

# %% load data
ds_2k = xr.open_dataset(
    "/work/bm1183/m301049/icon_hcap_data/plus2K/spinup/jed0003_atm_2d_tropical_mean.nc"
)
ds_2k_prod = xr.open_dataset(
    "/work/bm1183/m301049/icon_hcap_data/plus2K/production/jed0033_atm_2d_daymean_tropical_mean.nc"
)
ds_4k = xr.open_dataset(
    "/work/bm1183/m301049/icon_hcap_data/plus4K/spinup/jed0002_atm_2d_tropical_mean.nc"
)
ds_4K_prod = xr.open_dataset(
    "/work/bm1183/m301049/icon_hcap_data/plus4K/production/jed0022_atm_2d_daymean_tropical_mean.nc"
)
ds_control = xr.open_dataset(
    "/work/bm1183/m301049/icon_hcap_data/control/spinup/jed0001_atm_2d_daymean_tropical_mean.nc"
)
ds_control_prod = xr.open_dataset(
    "/work/bm1183/m301049/icon_hcap_data/control/production/jed0011_atm_2d_daymean_tropical_mean.nc"
)
ds_full = xr.open_dataset(
    "/work/bm1183/m301049/icon-mpim-2K/experiments/jed0033/jed0033_atm_2d_daymean_19790831T000000Z.15465121.nc"
)


# %% plot timeseries of all variables
def plot_var(varname, ax):
    ds_control[varname].isel(time=slice(1, None)).plot(
        ax=ax, label="control", color="black", linestyle="-."
    )
    ds_control_prod[varname].plot(ax=ax, label="control_prod", color="black")
    ds_2k[varname].plot(ax=ax, label="plus2K", color="orange", linestyle="-.")
    ds_2k_prod[varname].plot(ax=ax, label="plus2K_prod", color="orange")
    ds_4k[varname].plot(ax=ax, label="plus4K", color="red", linestyle="-.")
    ds_4K_prod[varname].plot(ax=ax, label="plus4K_prod", color="red")
    ax.set_title(ds_full[varname].attrs["long_name"])
    ax.set_ylabel(ds_full[varname].attrs["units"])


fig, axes = plt.subplots(8, 4, figsize=(30, 30), sharex="col")

for i, varname in enumerate(ds_full.data_vars):
    plot_var(varname, axes.flat[i])

fig.savefig('plots/spinup_tropical_mean_timeseries.png', dpi=300, bbox_inches='tight')

# %%
