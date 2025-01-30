# %%
import xarray as xr 
import os

# %%
run = 'jed0033'
exp_name = {'jed0011': 'control', 'jed0022': 'plus4K', 'jed0033': 'plus2K'}
path = f'/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/'
rand_coords = xr.open_dataset(f'{path}random_coords.nc')
filenames = [
    'atm_2d',
    'atm_3d_main',
    'atm_3d_rad',
    'atm_3d_cloud',
    'atm_3d_vel',
]
datasets = {}
idx_len = {}
for file in filenames:
        datasets[file] = xr.open_mfdataset(f'{path}{run}_{file}*.nc', combine='nested', concat_dim='index').sortby('index')
        list = [f for f in os.listdir(path) if f.startswith(f"{run}_{file}")]
        idx = 0
        for f in list:
            ds = xr.open_dataset(f'{path}{f}')
            idx = idx + len(ds.index)

        idx_len[file] = idx


# %% find missing indices 
missing_idx = rand_coords.index.values[~rand_coords.index.isin(datasets['atm_2d'].index.values)]
set(rand_coords.sel(index=missing_idx).time.values)


# %%
