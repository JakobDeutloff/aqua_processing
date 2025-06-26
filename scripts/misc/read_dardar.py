# %%
import os
import gzip
import shutil
import tempfile
import xarray as xr
import numpy as np
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# %%
path = '/work/um0878/data/dardar/DARDAR-CLOUD_v2.1.1/2008/*/*.hdf'
files = sorted(glob.glob(path))

# %%
def preprocess(file):
    ds = xr.open_dataset(file, engine='netcdf4')
    iwp = (ds['iwc'] * 60).sum('fakeDim44')
    ds_iwp = xr.Dataset(
        {
            'iwp': ('scanline', iwp.values),
            'latitude': ('scanline', ds.latitude.values),
            'longitude': ('scanline', ds.longitude.values),
        },
        coords={
            'scanline': np.arange(len(iwp)),
        }
    )
    return ds_iwp

with ProcessPoolExecutor() as executor:
    results = list(tqdm(executor.map(preprocess, files), total=len(files)))

# %%
ds = xr.concat(results, dim='scanline')
ds_concat = ds.assign_coords(scanline=np.arange(ds.sizes['scanline']))
ds_concat['iwp'] = ds_concat['iwp'] * 1e3
ds_concat['iwp'].attrs = {
    'name': 'ice water path',
    'unit': 'kg m^-2'
}

# %% save ds_concat
out_path = '/work/bm1183/m301049/dardar/'
ds_concat.to_netcdf(out_path + 'dardar_iwp_2008.nc')

# %%
