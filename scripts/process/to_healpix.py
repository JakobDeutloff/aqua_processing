# %%
import xarray as xr
from src.grid_helpers import merge_grid, to_healpix, fix_time
import pandas as pd
import sys

# %% load dataset
path = sys.argv[1]
save_path = sys.argv[2]
ds = (
    xr.open_mfdataset(
        path,
        chunks={"time": 1},
    )
    .pipe(merge_grid)
    .pipe(fix_time)
    .chunk(dict(ncells=-1))
)

# %% regrid to healpix
to_healpix(ds, save_path=save_path)
