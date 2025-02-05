# %% 
import os

# %%
filenames = [
    'atm_3d_rad_19',
    'atm_3d_cloud_19',
    'atm_3d_main_19',
    'atm_3d_vel_19',
    'atm_2d_19',
]

runs = [
    'jed0022',
    'jed0033',
    'jed0011',
]

path = os.getcwd()

for run in runs:
    for filename in filenames:
        os.system(f'sbatch {path}/scripts/subsample/submitter.sh {run} {filename}')