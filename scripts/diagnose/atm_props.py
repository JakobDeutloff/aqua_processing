# %%
import xarray as xr
from src.calc_variables import calc_connected

# %% load data
runs = ["jed0011", "jed0022", "jed0033"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
datasets = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_mil.nc"
    )
# %% calculate IWP and LWP 
for run in runs:
    datasets[run] = datasets[run].assign(iwp=datasets[run]['clivi'] + datasets[run]['qsvi'] + datasets[run]['qgvi'])
    datasets[run]['iwp'].attrs = {"long_name": "Ice Water Path", "units": "kg m^-2"}
    datasets[run] = datasets[run].assign(lwp=datasets[run]['cllvi'] + datasets[run]['qrvi'])
    datasets[run]['lwp'].attrs = {"long_name": "Liquid Water Path", "units": "kg m^-2"}

# %% calculate connectedness
for run in runs:
    datasets[run] = datasets[run].assign(conn=calc_connected(datasets[run]))


# %%
