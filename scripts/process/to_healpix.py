# %%
import xarray as xr
from src.grid_helpers import merge_grid, to_healpix


# %% load dataset
ds = (
    xr.open_dataset(
        "/work/bm1183/m301049/icon-mpim-2K/experiments/jed0033/jed0033_atm_2d_19790920T000000Z.14931256.nc",
        chunks={"time": 1},
    )
    .pipe(merge_grid)
    .chunk(dict(ncells=-1))
)
# %% regrid to healpix
to_healpix(ds, save_path="/work/bm1183/m301049/icon_hcap_data/plus2K/production/jed0033_atm_2d_19790920T000000Z.14931256_hp.nc")

