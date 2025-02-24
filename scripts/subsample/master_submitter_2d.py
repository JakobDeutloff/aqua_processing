# %% 
import os
import numpy as np

# %%
numbers = np.arange(6)
path = os.getcwd()

for number in numbers:
    os.system(f'sbatch {path}/scripts/subsample/submitter_2d.sh {number}')