# %%
from src.sampling import subsample_file
import os
import sys
import xarray as xr

# %% specify run and filetype
run = sys.argv[1]
filetype = sys.argv[2]
model_config = {
    "jed0011": "icon-mpim",
    "jed0022": "icon-mpim-4K",
    "jed0033": "icon-mpim-2K",
}
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
dir = f"/work/bm1183/m301049/{model_config[run]}/experiments/{run}"
filelist = [f for f in os.listdir(dir) if f.startswith(f"{run}_{filetype}")]

# %%  sample files
for file in filelist:
    print(f"subsample {file}")
    subsample_file(f"{dir}/{file}", exp_name[run])
# %% concatenate files 
path = (
    f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/"
)

# read ds from files
ds  = (
    xr.open_mfdataset(
        f"{path}{run}_{filetype}*.nc",
        combine="nested",
        concat_dim=["index"],
    )
    .sortby("index")
)
# save ds to file
datasets[file].to_netcdf(f"{path}{run}_{file[:-3]}_randsample.nc")
