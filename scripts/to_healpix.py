# %%
import healpix as hp
import xarray as xr

import easygems.remap as egr
from src.grid_helpers import merge_grid

# %% load dataset
ds = (
    xr.open_dataset(
        "/work/bm1183/m301049/icon-mpim/experiments/jed0001/jed0001_atm_2d_19790101T000000Z.14054703.nc",
        chunks={'time': 1},
    )
    .pipe(merge_grid)
    .chunk(dict(ncells=-1))
)

# %% load interpolation weigths
weights = xr.open_dataset("/home/m/m301049/aqua_processing/weights_z10.nc")
order = zoom = 10
nside = hp.order2nside(order)
npix = hp.nside2npix(nside)

# %% remap ds
print('remapping')
ds_remap = xr.apply_ufunc(
    egr.apply_weights,
    ds,
    kwargs=weights,
    keep_attrs=True,
    input_core_dims=[["ncells"]],
    output_core_dims=[["cell"]],
    output_dtypes=["f4"],
    vectorize=True,
    dask="parallelized",
    dask_gufunc_kwargs={
        "output_sizes": {"cell": npix},
    },
)
# %% attach grid metadata
ds_remap["crs"] = xr.DataArray(
    name="crs",
    data=[],
    dims="crs",
    attrs={
        "grid_mapping_name": "healpix",
        "healpix_nside": 2**zoom,
        "healpix_order": "nest",
    },
)

# %% save dataset to nc
print('saving')
ds_remap.to_netcdf(
    "/work/bm1183/m301049/icon-mpim/experiments/jed0001/jed0001_atm_2d_19790101T000000Z.14054703_remap.nc",
)
