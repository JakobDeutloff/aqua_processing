# %%
import sys
import xarray as xr
import numpy as np
import pickle as pkl
import os

# %%
run = sys.argv[1]
names = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}

iwp = xr.open_dataset(
    f"/work/bm1183/m301049/icon_hcap_data/{names[run]}/production/{run}_iwp.nc"
)["__xarray_dataarray_variable__"]

# %% calculate local time 
time_local= iwp.time.dt.hour + (iwp.clon / 15)
time_local = time_local.where(time_local < 24, time_local - 24)
time_local = time_local.where(time_local > 0, time_local + 24)
# %%
iwp_bins = np.logspace(-4, np.log10(40), 51)
iwp_slice  = iwp.where((time_local >= 13) & (time_local < 14), drop=True)
hist, edges = np.histogram(iwp_slice, bins=iwp_bins, density=False)
hist = hist / iwp_slice.size

path = f"/work/bm1183/m301049/icon_hcap_data/{names[run]}/production/iwp_hists"
if not os.path.exists(path):
    os.mkdir(path)
with open(
    f"{path}/{run}_iwp_hist_day.pkl",
    "wb",
) as f:
    pkl.dump(hist, f)

# %%
