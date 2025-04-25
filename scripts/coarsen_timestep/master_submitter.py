# %% 
import os
# %%
runs = [
    "jed0011",
    "jed0022",
    "jed0033",
]
files = [
    "atm_3d_main_19*.nc",
    "atm_3d_cloud_19*.nc",
    "atm_3d_rad_19*.nc",
    "atm_3d_vel_19*.nc",
]
model_config = {
    "jed0011": "icon-mpim",
    "jed0022": "icon-mpim-4K",
    "jed0033": "icon-mpim-2K",
}

for run in runs:
    for file in files:
        print(f"subsample run {run} file {file}")
        path = f"/work/bm1183/m301049/{model_config[run]}/experiments/{run}/"
        os.system(f'sbatch scripts/coarsen_timestep/submit_coarsening.sh {path} {run}_{file}')
    
 

