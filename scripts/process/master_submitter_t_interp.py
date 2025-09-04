# %% 
import os
# %%
runs = ["jed0011", "jed0022", "jed0033", "jed2224"]

for run in runs:
    os.system(f"sbatch scripts/process/submitter_t_interp.sh {run}")