# %%
import xarray as xr
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.units import units
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import sys

# %% load CRE data
run = sys.argv[1]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
cre_interp_mean = {}
cre_arr = {}
cre_arr_interp = {}
datasets = {}
datasets[run] = xr.open_dataset(
    f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed.nc"
)

#%% Define a function to compute CAPE and CIN for a single profile
def compute_cape_cin(index):
    cape_i, cin_i = mpcalc.surface_based_cape_cin(
        p.isel(index=index), t.isel(index=index), dew_point.isel(index=index)
    )
    return cape_i, cin_i

# %% calculate CAPE
print(run)
# prepare data
p = datasets[run]["pfull"].isel(height=slice(None, None, -1)) * units.Pa
t = datasets[run]["ta"].isel(height=slice(None, None, -1)) * units.degK
q = datasets[run]["hus"].isel(height=slice(None, None, -1)) * units.dimensionless

# calculate dewpoint
dew_point = mpcalc.dewpoint_from_specific_humidity(
p, t, q
)

# Parallelize the loop
cape = []
cin = []
with ProcessPoolExecutor() as executor:
    futures = {executor.submit(compute_cape_cin, i): i for i in range(len(p))}
    for future in tqdm(as_completed(futures), total=len(futures)):
        cape_i, cin_i = future.result()
        cape.append(cape_i.magnitude)
        cin.append(cin_i.magnitude)

#  build xarrays 
cape_cin_ds = xr.Dataset(
    {
        "cape": xr.DataArray(
            cape,
            dims=["index"],
            coords={"index": datasets["jed0011"]["index"]},
        ),
        "cin": xr.DataArray(
            cin,
            dims=["index"],
            coords={"index": datasets["jed0011"]["index"]},
        ),
    }
)
cape_cin_ds['cape'].attrs = {
    "long_name": "CAPE",
    "units": "J/kg",
    "description": "Convective Available Potential Energy",
}
cape_cin_ds['cin'].attrs = {
    "long_name": "CIN",
    "units": "J/kg",
    "description": "Convective Inhibition Energy",
}

# save file 
path = f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_cape_cin.nc"
cape_cin_ds.to_netcdf(path)
