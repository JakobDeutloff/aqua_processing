# %%
import xarray as xr
from src.grid_helpers import merge_grid, to_healpix


# %% load dataset
ds = (
    xr.open_mfdataset(
        "/work/bm1183/m301049/icon-mpim-2K/experiments/jed0003/jed0003_atm_2d_daymean_*.nc",
        chunks={"time": 1},
    )
    .pipe(merge_grid)
    .chunk(dict(ncells=-1))
)
# %% regrid to healpix
to_healpix(ds, save_path="/work/bm1183/m301049/icon-hcap/plus2K/jed0003_atm_2d_daymean_hp.nc")

