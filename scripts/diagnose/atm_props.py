# %%
import xarray as xr
from src.calc_variables import (
    calc_connected,
    calculate_hc_temperature_bright,
)
import os
from dask.diagnostics import ProgressBar
import dask

# %% load data
runs = ["jed0011", "jed0022", "jed0033", "jed2224"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K", "jed2224": "const_o3"}
datasets = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_2.nc", chunks={'index':100000}
    )

vgrid = (
    xr.open_dataset(
        "/work/bu1562/m301049/icon-mpim/experiments/jed0001/atm_vgrid_angel.nc"
    )
    .mean("ncells")
    .rename({"height_2": "height", "height": "height_2"})
)
# %% calculate all derived variables in a single pass
for run in runs:
    ds = datasets[run]

    # IWP and LWP
    ds = ds.assign(iwp=ds["clivi"] + ds["qsvi"] + ds["qgvi"])
    ds["iwp"].attrs = {"long_name": "Ice Water Path", "units": "kg m^-2"}
    ds = ds.assign(lwp=ds["cllvi"] + ds["qrvi"])
    ds["lwp"].attrs = {"long_name": "Liquid Water Path", "units": "kg m^-2"}

    # connectedness
    ds = ds.assign(conn=calc_connected(ds, vgrid["zg"]))

    # cloud top
    ds["hc_top_temperature"], ds["hc_top_pressure"] = (
        calculate_hc_temperature_bright(ds, vgrid["zg"])
    )

    # masks
    ds["mask_hc_no_lc"] = (ds["iwp"] > 1e-5) & (ds["lwp"] < 1e-4)
    ds["mask_low_cloud"] = ((ds["conn"] == 0) & (ds["lwp"] > 1e-4)) * 1

    # local time
    ds = ds.assign(time_local=lambda d: d.time.dt.hour + (d.clon / 15))
    ds["time_local"] = (
        ds["time_local"]
        .where(ds["time_local"] < 24, ds["time_local"] - 24)
        .where(ds["time_local"] > 0, ds["time_local"] + 24)
    )
    ds["time_local"].attrs = {"units": "h", "long_name": "Local time"}

    datasets[run] = ds

# %% save processed data as float64 — write all runs in parallel
paths = {
    run: f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed_64_2.nc"
    for run in runs
}
for run in runs:
    if os.path.exists(paths[run]):
        os.remove(paths[run])

delayed_writes = [
    datasets[run].astype(float).to_netcdf(paths[run], engine="h5netcdf", compute=False)
    for run in runs
]
print(f"Saving processed data for all runs in parallel...")
with ProgressBar():
    dask.compute(*delayed_writes)

# %%
