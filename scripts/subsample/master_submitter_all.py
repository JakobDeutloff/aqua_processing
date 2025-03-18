# %% 
import os
from src.sampling import get_random_coords

# %%
runs = [
    "jed0011",
    "jed0022",
    "jed0033",
]

files = [
    "atm_2d_19",
    "atm_3d_main_19",
    "atm_3d_cloud_19",
    "atm_3d_rad_19",
    "atm_3d_vel_19",
]

model_config = {
    "jed0011": "icon-mpim",
    "jed0022": "icon-mpim-4K",
    "jed0033": "icon-mpim-2K",
}

exp_name = {
    "jed0011": "control",
    "jed0022": "plus4K",
    "jed0033": "plus2K",
}

for run in runs:
    print(f"get random coords for {run}")
    get_random_coords(run, model_config[run], exp_name[run], number=1)
    
    for file in files:
        os.system(f'sbatch scripts/subsample/submitter_all.sh {run} {file}')

# %%
