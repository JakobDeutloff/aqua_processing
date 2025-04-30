# %%
from src.sampling import subsample_file, concatenate_files, merge_files
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
followups = {
    "jed0011": "jed0111",
    "jed0022": "jed0222",
    "jed0033": "jed0333",
}
dir1 = f"/work/bm1183/m301049/{model_config[run]}/experiments/{run}"
dir2 = f"/work/bm1183/m301049/{model_config[run]}/experiments/{followups[run]}"
filelist1 = [f for f in os.listdir(dir1) if f.startswith(f"{run}_{filetype}")]
filelist2 = [f for f in os.listdir(dir2) if f.startswith(f"{followups[run]}_{filetype}")]


#%%  sample files
for file in filelist1:
    print(f"subsample {file}")
    subsample_file(f"{dir1}/{file}", exp_name[run], 0)

for file in filelist2:
    print(f"subsample {file}")
    subsample_file(f"{dir2}/{file}", exp_name[run], 0)

# %% concatenate files
concatenate_files(run, followups[run], exp_name[run], filetype)

# %%
