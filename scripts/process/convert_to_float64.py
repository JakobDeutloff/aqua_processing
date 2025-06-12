# %%
from src.read_data import load_definitions
import xarray as xr

# Only load definitions, not all datasets at once
runs, exp_name, colors, labels = load_definitions()

path = "/work/bm1183/m301049/icon_hcap_data/"
for run in runs:
    print(run)
    # Open one dataset at a time, with dask chunks for efficiency
    ds = xr.open_dataset(
        path + f"{exp_name[run]}/production/random_sample/{run}_randsample_processed.nc",
        chunks={}
    )
    ds64 = ds.astype(float)
    ds64.to_netcdf(path + f"{exp_name[run]}/production/random_sample/{run}_randsample_processed_64.nc")
    ds.close()
    ds64.close()