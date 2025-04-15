# %%
import xarray as xr
import easygems.healpix as egh
import os

# %%
runs = ["jed0011", "jed0033", "jed0022"]
spinups = {"jed0011": "jed0001", "jed0022": "jed0002", "jed0033": "jed0003"}
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
colors = {"jed0011": "k", "jed0022": "r", "jed0033": "orange"}
datasets = {}

# %%
for run in runs:
    savepath = f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/{run}_atm_2d_daymean_hp.nc"
    if os.path.exists(savepath):
        os.remove(savepath)
    print(run)
    xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/{run}_atm_2d_hp.nc"
    ).pipe(egh.attach_coords).groupby(
        "time.day"
    ).mean(
        dim="time"
    ).to_netcdf(savepath
    )

# %%
