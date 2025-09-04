# %%
import xarray as xr
from src.grid_helpers import merge_grid, fix_time, to_healpix

# %% load data
run = 'jed0011'
configs = {"jed0011": "icon-mpim", "jed0022": "icon-mpim-4K", "jed0033": "icon-mpim-2K"}
names = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}

ds = (
    xr.open_mfdataset(
        f"/work/bm1183/m301049/icon-mpim/experiments/jed0011/jed0011_atm_2d_19*.nc"
    )
    .pipe(merge_grid)
    .pipe(fix_time)[["clivi", "qsvi", "qgvi", "qrvi", "cllvi"]]
)
# %% calculate IWP
twp = (ds["clivi"] + ds["qsvi"] + ds["qgvi"] + ds["qrvi"] + ds["cllvi"])
twp.attrs = {
    'long_name': 'total water path',
    'unit': 'kg m**-2',
}
twp = twp.chunk(dict(ncells=-1, time=1))
twp = xr.Dataset({'twp': twp})

# %% interpolate to healpix and save
twp_hp = to_healpix(twp, '/work/bm1183/m301049/icon_hcap_data/control/production/twp_hp.nc')

# %%
