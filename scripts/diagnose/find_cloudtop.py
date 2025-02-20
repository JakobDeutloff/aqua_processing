# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# %% load data
runs = ["jed0011"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
datasets = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed.nc"
    )

# %% calculate brightness temp from fluxes
def calc_brightness_temp(flux):
    return (flux / 5.67e-8) ** (1 / 4)


#%% only look at clouds with cloud tops above 350 hPa and IWP > 1e-1 so that e = 1 can be assumed
atms = datasets["jed0011"]
mask = (atms["iwp"] > 1e-1) 

flux = atms["rlut"]
T_bright = calc_brightness_temp(flux)
T_bright = T_bright.where(mask, 500)

#%% tropopause pressure
height_trop_ixd = atms["ta"].argmin("height")
p_trop = atms["phalf"].isel(height=height_trop_ixd)

# get pressure levels and ice_cumsum where T == T_bright in Troposphere
T_profile = atms["ta"].where(atms["phalf"] > p_trop, 0)
height_bright_idx = np.abs(T_profile - T_bright).argmin("height")  # fills with 0 for NaNs
ice_cumsum = atms["iwc_cumsum"].isel(height=height_bright_idx)
ice_cumsum = ice_cumsum.where(mask)
p_bright = atms.isel(height=height_bright_idx)["phalf"].where(mask)
T_bright = T_bright.where(mask)
mean_ice_cumsum = ice_cumsum.median()
print(mean_ice_cumsum.values)

