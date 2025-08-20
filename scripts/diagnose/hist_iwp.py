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
# %%
timeslices = {"30_days": slice(0, 30*24), "60_days": slice(0, 60*24), "full": slice(0, None)}
iwp_bins = np.logspace(-4, np.log10(40), 51)
for timeslice in timeslices.keys():
    print(timeslice)
    iwp_slice = iwp.isel(time=timeslices[timeslice])
    hist, edges = np.histogram(iwp_slice, bins=iwp_bins, density=False)
    hist = hist / iwp_slice.size

    path = f"/work/bm1183/m301049/icon_hcap_data/{names[run]}/production/iwp_hists"
    if not os.path.exists(path):
        os.mkdir(path)
    with open(
        f"{path}/{run}_iwp_hist{timeslice}.pkl",
        "wb",
    ) as f:
        pkl.dump(hist, f)
