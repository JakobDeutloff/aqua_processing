# %%
import xarray as xr 
import os
import glob
import re

# %%
runs = ['jed0011', 'jed0033']
exp_name = {'jed0011': 'control', 'jed0022': 'plus4K', 'jed0033': 'plus2K'}
filenames = [
    'atm_2d',
    'atm_3d_main',
    'atm_3d_rad',
    'atm_3d_cloud',
    'atm_3d_vel',
]

datasets = {}
for run in runs:
    path = f'/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/'
    rand_coords = xr.open_dataset(f'{path}random_coords.nc')
    for file in filenames:
            # read ds from files
            datasets[file] = xr.open_mfdataset(f'{path}{run}_{file}*.nc', combine='nested', concat_dim=['index']).drop_duplicates('index').sortby('index')
            # check for missing indices 
            missing_idx = rand_coords.index.values[~rand_coords.index.isin(datasets[file].index.values)]
            print(f'Missing indices for {run} {file}: {missing_idx}')
            # save ds as one file
            if len(missing_idx)==0:
                if os.path.exists(f'{path}{run}_{file}_randsample.nc'):
                    os.remove(f'{path}{run}_{file}_randsample.nc')
                datasets[file].to_netcdf(f'{path}{run}_{file}_randsample.nc')
                os.system(f'rm {path}{run}_{file}_19*.nc')
                


# %%
