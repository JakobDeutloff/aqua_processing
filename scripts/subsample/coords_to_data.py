# %%
from src.sampling import subsample_file
import os
import sys

# %% get list of all files
run = sys.argv[1]
filetype = sys.argv[2]
print(f"Processing {run} for {filetype}")
model_config = {
    "jed0011": "icon-mpim",
    "jed0022": "icon-mpim-4K",
    "jed0033": "icon-mpim-2K",
}
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
dir = f"/work/bm1183/m301049/{model_config[run]}/experiments/{run}/"
filelist = [f for f in os.listdir(dir) if f.startswith(f"{run}_{filetype}")]
print("Files to process:", filelist)
# %%
for file in filelist:
    print(f"Processing {file}")
    subsample_file(f"{dir}/{file}", exp_name[run])

# %%
