# %%
import xarray as xr
from src.calc_variables import (
    calc_connected,
)
import os
from concurrent.futures import ProcessPoolExecutor

# %% load data
runs = ["jed0011", "jed0033", "jed0022"]
exp_name = {
    "jed0011": "control",
    "jed0022": "plus4K",
    "jed0033": "plus2K",
    "jed2224": "const_o3",
}
datasets = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed_64.nc",
    )

vgrid = (
    xr.open_dataset(
        "/work/bu1562/m301049/icon-mpim/experiments/jed0001/atm_vgrid_angel.nc"
    )
    .mean("ncells")
    .rename({"height_2": "height", "height": "height_2"})
)


# %% calculate connectedness
def process_connectedness(args):
    """Helper function for parallel processing"""
    run, ds, vgrid_zg, frac = args
    return run, ds.assign(conn=calc_connected(ds, vgrid_zg, frac_no_cloud=frac))


#  calculate connectedness
with ProcessPoolExecutor(max_workers=3) as executor:
    args_list = [(run, datasets[run], vgrid["zg"], 0.01) for run in runs]
    results = list(executor.map(process_connectedness, args_list))

# Update datasets dictionary with results
for run, ds in results:
    datasets[run] = ds
    print(f"Completed {run}")

# %% save connectedness
for run in runs:
    path = f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_connectedness_1.nc"
    if os.path.exists(path):
        os.remove(path)
    datasets[run]['conn'].to_netcdf(path)

# %%
