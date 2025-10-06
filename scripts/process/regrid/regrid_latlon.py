# %%
import os
import xarray as xr
from src.grid_helpers import fix_time
import sys

# %%
run = sys.argv[1]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
configs = {
    "jed0011": "icon-mpim",
    "jed0022": "icon-mpim-4K",
    "jed0033": "icon-mpim-2K",
}
followups = {
    "jed0011": "jed0111",
    "jed0022": "jed0222",
    "jed0033": "jed0333",
}

# %% concatenate datasets
print(f"Processing {run}...")
path = f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/latlon/atm3d_icon.nc"

print("Concatenating datasets...")
ds_first_month = (
    xr.open_mfdataset(
        f"/work/bm1183/m301049/{configs[run]}/experiments/{run}/{run}_atm_3d_*.nc"
    )
    .pipe(fix_time)[['hus', 'ta', 'ua', 'va']]
)
ds_last_two_months = (
    xr.open_mfdataset(
        f"/work/bm1183/m301049/{configs[run]}/experiments/{followups[run]}/{followups[run]}_atm_3d_19*.nc"
    )
    .pipe(fix_time)[['hus', 'ta', 'va', 'rho']]
)
ds = xr.concat([ds_first_month, ds_last_two_months], dim="time").astype(float)
# select one value per day 
ds = ds.sel(time=(ds.time.dt.minute == 0) & (ds.time.dt.hour == 0))
ds.to_netcdf(path)

# %% regrid 2D datasets
path = f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/latlon"
os.system(f"sbatch scripts/process/regrid/regrid_3d.sh {path}")
