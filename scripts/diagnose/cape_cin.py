import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import sys
import os

# %% Load CRE data
run = sys.argv[1]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
ds = xr.open_dataset(
    f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed.nc"
)

# Prepare data
p = ds["pfull"].isel(height=slice(None, None, -1)) * units.Pa
t = ds["ta"].isel(height=slice(None, None, -1)) * units.degK
q = ds["hus"].isel(height=slice(None, None, -1)) * units.dimensionless

# Calculate dewpoint
dew_point = mpcalc.dewpoint_from_specific_humidity(p, t, q)

# Function to compute CAPE and CIN for a chunk
def compute_cape_cin_chunk(start, end):
    cape_chunk = []
    cin_chunk = []
    p_chunk = p.isel(index=slice(start, end))
    t_chunk = t.isel(index=slice(start, end))
    dew_point_chunk = dew_point.isel(index=slice(start, end))
    for i in range(len(p_chunk.index)):
        try:
            cape_i, cin_i = mpcalc.surface_based_cape_cin(
                p_chunk.isel(index=i), t_chunk.isel(index=i), dew_point_chunk.isel(index=i)
            )
            cape_chunk.append(cape_i.magnitude)
            cin_chunk.append(cin_i.magnitude)
        except ValueError:
            cape_chunk.append(None)
            cin_chunk.append(None)
    return cape_chunk, cin_chunk

# Parallelize chunk processing
chunk_size = 10000  # Adjust chunk size based on memory and performance
num_workers = int(os.cpu_count() * 0.75)  # Use 75% of available CPU cores
total_profiles = len(p.index)

cape = []
cin = []

with ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = []
    for i in range(0, total_profiles, chunk_size):
        end = min(i + chunk_size, total_profiles)
        futures.append(executor.submit(compute_cape_cin_chunk, i, end))
    
    for future in tqdm(futures, total=len(futures)):
        cape_chunk, cin_chunk = future.result()
        cape.extend(cape_chunk)
        cin.extend(cin_chunk)

# Build Xarray Dataset
cape_cin_ds = xr.Dataset(
    {
        "cape": xr.DataArray(
            cape,
            dims=["index"],
            coords={"index": ds["index"]},
        ),
        "cin": xr.DataArray(
            cin,
            dims=["index"],
            coords={"index": ds["index"]},
        ),
    }
)
cape_cin_ds["cape"].attrs = {
    "long_name": "CAPE",
    "units": "J/kg",
    "description": "Convective Available Potential Energy",
}
cape_cin_ds["cin"].attrs = {
    "long_name": "CIN",
    "units": "J/kg",
    "description": "Convective Inhibition Energy",
}

# Save to NetCDF
output_path = f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_cape_cin.nc"
cape_cin_ds.to_netcdf(output_path)