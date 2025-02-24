# %% 
import os
import numpy as np

# %%
runs = [
    "jed0011",
    "jed0022",
    "jed0033",
]

files = [
    "atm_2d_19*.nc",
    "atm_3d_main_19*.nc",
    "atm_3d_cloud_19*.nc",
    "atm_3d_rad_19*.nc",
    "atm_3d_vel_19*.nc",
]

for run in runs:
    for file in files:
        os.system(f'python scripts/subsample/sample_all.py {run} {file}')
