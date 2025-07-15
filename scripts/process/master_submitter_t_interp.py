# %% 
import os
# %%
runs = ["jed2224"]

for run in runs:
    os.system(f"sbatch scripts/process/submitter_t_interp.sh {run}")