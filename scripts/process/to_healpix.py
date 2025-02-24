# %%
import xarray as xr
from src.grid_helpers import merge_grid, to_healpix, fix_time
import pandas as pd

# %% load dataset
ds = (
    xr.open_mfdataset(
        "/work/bm1183/m301049/icon-mpim/experiments/jed0011/jed0011_atm_2d_19*.nc",
        chunks={"time": 1},
    )
    .pipe(merge_grid)
    .pipe(fix_time)
    .chunk(dict(ncells=-1))
)
# %% coarsen to 6h timestep
time_coarse = pd.date_range(start=ds.time.values[0], end=ds.time.values[-1], freq='6h')
ds_coarse = ds.sel(time=time_coarse)

# %% regrid to healpix
to_healpix(ds_coarse, save_path="/work/bm1183/m301049/icon_hcap_data/control/production/jed0011_atm_2d_hp.nc")
