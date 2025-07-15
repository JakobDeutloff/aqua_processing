# %% import
import os
from src.sampling import merge_files

# %% set names
runs = ["jed0011", "jed0022", "jed0033", "jed2224"]
followups = {
    "jed0011": "jed0111",
    "jed0022": "jed0222",
    "jed0033": "jed0333",
    "jed2224": None,
}
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K", "jed2224": "const_o3"}
filenames = [
    "atm_2d_19",
    "atm_3d_main_19",
    "atm_3d_rad_19",
    "atm_3d_cloud_19",
    "atm_3d_vel_19",
]

# %% merge data
for run in runs:
    print(run)
    merge_files(
        run, filenames, exp_name[run]
    )
# %% delete single files 
for run in runs:
    for file in filenames:
        path = f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/"
        files = [f for f in os.listdir(path) if (f.startswith(f"{run}_{file}")) or (f.startswith(f"{followups[run]}_{file}"))]
        for file in files:
            os.remove(path+file)
# %%
