# %%
from src.sampling import get_coarse_time, get_ds_list, coarsen_ds
import sys 

# %%
path = sys.argv[1]
pattern = sys.argv[2]

# %% get list of all datasets in path matching pattern
ds_list = get_ds_list(path, pattern)

# %% construct coarser time axis 
time_coarse = get_coarse_time(ds_list)

# %% coarsen datasets in ds_list to new time axis
coarsen_ds(ds_list, time_coarse)

# %%
