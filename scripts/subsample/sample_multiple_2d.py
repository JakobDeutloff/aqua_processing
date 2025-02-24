# %%
from src.sampling import subsample_file, get_random_coords
import sys

# %% specify run and filetype
run = "jed0011" 
filetype = "atm_2d_19*.nc" 
number = sys.argv[1]
model_config = {
    "jed0011": "icon-mpim",
    "jed0022": "icon-mpim-4K",
    "jed0033": "icon-mpim-2K",
}
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
dir = f"/work/bm1183/m301049/{model_config[run]}/experiments/{run}"

#%%  sample files
print(f"get random coords for {run}")
get_random_coords(run, model_config[run], exp_name[run], number)
print(f"subsample files for {run}")
subsample_file(f"{dir}/{run}_{filetype}", exp_name[run], number)

# %%
