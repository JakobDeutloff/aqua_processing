# %%
import xarray as xr
import os
from src.calc_variables import calc_heating_rates, calc_stability

# %% load data
runs = ["jed0011", "jed0022", "jed0033"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
datasets = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample.nc"
    )

vgrid = (
    xr.open_dataset(
        "/work/bm1183/m301049/icon-mpim/experiments/jed0001/atm_vgrid_angel.nc"
    )
    .mean("ncells")
    .rename({"height": "height_2", "height_2": "height"})
)


# %% calculate heating rates
hrs = {}
for run in runs:
    print(run)
    hrs[run] = calc_heating_rates(
        datasets[run]["rho"],
        datasets[run]["rsd"] - datasets[run]["rsu"],
        datasets[run]["rld"] - datasets[run]["rlu"],
        vgrid,
    )
    hrs[run] = hrs[run].assign(
        stab=calc_stability(datasets[run]["ta"], vgrid=vgrid)
    )

    # save data
    path = f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_heating_rates.nc"
    if os.path.exists(path):
        os.remove(path)
    hrs[run].to_netcdf(path)
        
