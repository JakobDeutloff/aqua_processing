# %% import 
import os 

# %%
runs = ['jed0011', 'jed0022', 'jed0033']
for run in runs:
    os.system(f'sbatch scripts/diagnose/submit.sh {run}')
# %%
