# %%
import xarray as xr
from src.calc_variables import (
    calc_connected,
    calc_IWC_cumsum,
    calculate_hc_temperature,
)
import os

# %% load data
runs = ["jed0011"] #["jed0011", "jed0022", "jed0033"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
datasets = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_mil.nc"
    )

vgrid = xr.open_dataset("/work/bm1183/m301049/icon-mpim/experiments/jed0001/atm_vgrid_angel.nc")
# %% calculate IWP and LWP 
for run in runs:
    datasets[run] = datasets[run].assign(
        iwp=datasets[run]["clivi"] + datasets[run]["qsvi"] + datasets[run]["qgvi"]
    )
    datasets[run]["iwp"].attrs = {"long_name": "Ice Water Path", "units": "kg m^-2"}
    datasets[run] = datasets[run].assign(
        lwp=datasets[run]["cllvi"] + datasets[run]["qrvi"]
    )
    datasets[run]["lwp"].attrs = {"long_name": "Liquid Water Path", "units": "kg m^-2"}

# %% calculate cumsum of IWC
for run in runs:
    datasets[run] = datasets[run].assign(dzghalf=vgrid["dzghalf"].mean("ncells"))
    datasets[run] = datasets[run].assign(iwc_cumsum=calc_IWC_cumsum(datasets[run]))

# %% calculate connectedness
for run in runs:
    if "conn" not in datasets[run]:
        datasets[run] = datasets[run].assign(conn=calc_connected(datasets[run]))
# %% calculate cloud top
for run in runs:
    datasets[run]["hc_top_temperature"], datasets[run]["hc_top_pressure"] = (
        calculate_hc_temperature(datasets[run], IWP_emission=8.17e-3)
    )
# %% calculate masks
for run in runs:
    datasets[run]["mask_height"] = datasets[run]["hc_top_pressure"] < 350
    datasets[run]["mask_hc_no_lc"] = (datasets[run]["iwp"] > 1e-5) & (
        datasets[run]["lwp"] < 1e-4
    )
    datasets[run]["mask_low_cloud"] = (
        (datasets[run]["conn"] == 0) & (datasets[run]["lwp"] > 1e-4)
    ) * 1

# %% assign local time
for run in runs:
    datasets[run] = datasets[run].assign(time_local=lambda d: d.time.dt.hour + d.clon / 15)
    datasets[run]["time_local"] = datasets[run]["time_local"].where(
        datasets[run]["time_local"] < 24, datasets[run]["time_local"] - 24
    )
    datasets[run]["time_local"].attrs = {"units": "h", "long_name": "Local time"}

# %% save processed data
for run in runs:
    path = f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed.nc"
    if os.path.exists(path):
        os.remove(path)
    datasets[run].to_netcdf(path)

# %%
