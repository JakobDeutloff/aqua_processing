# %%
from src.coarsening import get_coarse_time, get_ds_list, coarsen_ds

# %%
path = "/work/bm1183/m301049/icon-mpim/experiments/jed0011/"
first = "jed0011_atm_3d_vel_19790630T000000Z.14357926.nc"
last = "jed0011_atm_3d_vel_19790730T000000Z.14380341.nc"
pattern = "jed0011_atm_3d_vel_1979*"

# %% construct coarser time axis 
time_coarse = get_coarse_time(first, last, path)

# %% get list of all datasets in path matching pattern
ds_list = get_ds_list(path, pattern)

# %% coarsen datasets in ds_list to new time axis
coarsen_ds(ds_list, time_coarse)
