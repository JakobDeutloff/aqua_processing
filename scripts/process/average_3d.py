# %%
import xarray as xr
from src.grid_helpers import merge_grid, fix_time
from dask.diagnostics import ProgressBar
import os

# %%
print("load first part")
ozone_control_1 = (
    xr.open_mfdataset(
        "/work/bm1183/m301049/icon-mpim/experiments/jed0011/jed0011_atm_3d_main_19*.nc",
        chunks={},
    )
    .pipe(merge_grid)
    .pipe(fix_time)[["o3", "pfull"]]
    .astype(float)
)
print("load second part")
ozone_control_2 = (
    xr.open_mfdataset(
        "/work/bm1183/m301049/icon-mpim/experiments/jed0111/jed0111_atm_3d_main_19*.nc",
        chunks={},
    )
    .pipe(merge_grid)
    .pipe(fix_time)[["o3", "pfull"]]
).astype(float)
print("concatenate")
ozone_control = xr.concat([ozone_control_1, ozone_control_2], dim="time")
print("subselect 1 timestep every week")
ozone_control = ozone_control.isel(time=slice(0, None, 7))
print("convert to mol/mol")
mol_weight_o3 = 48  # g/mol
mol_weight_dry_air = 28.97  # g/mol
ozone_control["o3"] = ozone_control["o3"] * (mol_weight_dry_air / mol_weight_o3)
print("calculate mean")
ozone_control = ozone_control.mean(dim="time")

print("save")
path = "/work/bm1183/m301049/icon_hcap_data/control/production/ozone_control.nc"
if os.path.exists(path):
    print(f"File {path} already exists. Deleting it.")
    os.remove(path)
with ProgressBar():
    ozone_control.to_netcdf(
        "/work/bm1183/m301049/icon_hcap_data/control/production/ozone_control.nc"
    )
