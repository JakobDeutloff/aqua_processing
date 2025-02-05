# %% import
import xarray as xr
import os

# %% set names
runs = ["jed0011", "jed0022", "jed0033"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
filenames = [
    "atm_2d_19",
    "atm_3d_main_19",
    "atm_3d_rad_19",
    "atm_3d_cloud_19",
    "atm_3d_vel19",
]

# %% concatenate single files
datasets = {}
for run in runs:
    path = (
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/"
    )
    rand_coords = xr.open_dataset(f"{path}random_coords.nc")
    for file in filenames:
        # read ds from files
        datasets[file] = (
            xr.open_mfdataset(
                f"{path}{run}_{file}*.nc",
                combine="nested",
                concat_dim=["index"],
                coords="minimal",
            )
            .drop_duplicates("index")
            .sortby("index")
        )
        # check for missing indices
        missing_idx = rand_coords.index.values[
            ~rand_coords.index.isin(datasets[file].index.values)
        ]
        print(f"Missing indices for {run} {file[:-3]}: {missing_idx}")
        # save ds as one file
        if len(missing_idx) == 0:
            if os.path.exists(f"{path}{run}_{file[:-3]}_randsample_mil.nc"):
                os.remove(f"{path}{run}_{file[:-3]}_randsample_mil.nc")
            datasets[file].to_netcdf(f"{path}{run}_{file[:-3]}_randsample_mil.nc")
            # os.system(f"rm {path}{run}_{file}_19*.nc")


# %% merge 2D and 3D data
for run in runs:
    datasets = []
    for file in filenames:
        datasets.append(
            xr.open_dataset(
                f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_{file[:-3]}_randsample_mil.nc"
            )
        )
    datasets[0] = datasets[0].rename({"height_2": "s_height_2", "height": "s_height"})
    ds = xr.merge(datasets)
    path = f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_mil.nc"
    if os.path.exists(path):
        os.remove(path)
    ds.to_netcdf(path)

# %%
