import xarray as xr
import pandas as pd
from dask.diagnostics import ProgressBar
import glob
import os
from src.grid_helpers import merge_grid, fix_time
import numpy as np
from tqdm import tqdm
import gc
import dask.array as da


def get_coarse_time(list):
    """
    Build new time axis with 6h frequency
    """
    ds_first = xr.open_dataset(list[0], chunks={})
    ds_last = xr.open_dataset(list[-1], chunks={})
    time = pd.date_range(
        start=ds_first.isel(time=0).time.values,
        end=ds_last.isel(time=-1).time.values,
        freq="1d",
    )
    return time


def get_ds_list(path, pattern):
    """
    Get list of all datasets in path matching pattern
    """
    ds_list = glob.glob(path + pattern)
    ds_list.sort()
    return ds_list


def coarsen_ds(ds_list, time):
    """
    Coarsen datasets in ds_list to new time axis
    """
    for file in ds_list:
        print("Processing " + file)
        ds = xr.open_dataset(file, chunks={})
        time_subset = time[(time >= ds.time.values[0]) & (time <= ds.time.values[-1])]
        ds_coarse = ds.sel(time=time_subset, method="nearest")
        with ProgressBar():
            ds_coarse.to_netcdf(file[:-3] + "_coarse.nc")
            os.remove(file)
            os.rename(file[:-3] + "_coarse.nc", file)


def subsample_file(file, exp_name, number=0):
    """
    Get random sample of data from file
    """

    ds = (
        xr.open_dataset(
            file,
            chunks={"height": -1},
        )
        .pipe(merge_grid)
        .pipe(fix_time)
    )

    random_coords = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name}/production/random_sample/random_coords{number}.nc"
    ).load()

    # get overlap
    valid_time_mask = random_coords.time.isin(ds.time)
    valid_coords = random_coords.where(valid_time_mask, drop=True)

    # select data
    print("Selecting data")
    ds_random = ds.sel(
        ncells=valid_coords.ncells.astype(int), time=valid_coords.time
    ).assign_coords(
        time=valid_coords.time,
        ncells=valid_coords.ncells,
        clat=ds.clat.sel(ncells=valid_coords.ncells),
        clon=ds.clon.sel(ncells=valid_coords.ncells),
    )

    # save in random sample folder
    print("Saving data")
    filename = file.split("/")[-1]
    path = f"/work/bm1183/m301049/icon_hcap_data/{exp_name}/production/random_sample/{filename}"
    if os.path.exists(path):
        os.remove(path)
    with ProgressBar():
        ds_random.to_netcdf(path)

    # clear RAM
    del ds
    del ds_random
    del valid_coords
    del random_coords
    del valid_time_mask
    gc.collect()


def get_random_coords(run, followup, model_config, exp_name, number=0):

    path = f"/work/bm1183/m301049/{model_config}/experiments/{run}/"

    # check if random coords for that number already exist
    if os.path.exists(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name}/production/random_sample/random_coords{number}.nc"
    ):
        print("Random coords already exist")
        return
    
    ds_first = (
        xr.open_mfdataset(f"{path}{run}_atm_3d_main_19*.nc", chunks={})
        .pipe(merge_grid)
        .pipe(fix_time)
    )
    if followup is None:
        ds = ds_first
    else:
        path = f"/work/bm1183/m301049/{model_config}/experiments/{followup}/"
        ds_second = (
            xr.open_mfdataset(f"{path}{followup}_atm_3d_main_19*.nc", chunks={})
            .pipe(merge_grid)
            .pipe(fix_time)
        )
        ds = xr.concat([ds_first, ds_second], dim="time")

    # select tropics
    ds_trop = ds.where((ds.clat < 20) & (ds.clat > -20), drop=True)

    # get random coordinates across time and ncells
    ncells = ds_trop.sizes["ncells"]
    time = ds_trop.sizes["time"]

    # Generate unique pairs of random indices
    num_samples = int(1e7)
    total_indices = ncells * time
    random_indices = da.random.randint(0, total_indices, num_samples).compute()

    random_ncells_idx = random_indices % ncells
    random_time_idx = random_indices // ncells

    # create xrrays
    random_coords = xr.Dataset(
        {
            "time": xr.DataArray(ds_trop.time[random_time_idx].values, dims="index"),
            "ncells": xr.DataArray(
                ds_trop.ncells[random_ncells_idx].values, dims="index"
            ),
        },
        coords={"index": np.arange(num_samples)},
    )

    #  save to file
    save_path = f"/work/bm1183/m301049/icon_hcap_data/{exp_name}/production/random_sample/random_coords{number}.nc"
    if os.path.exists(save_path):
        os.remove(save_path)

    random_coords.to_netcdf(save_path)


def process_bin(i, j, ds, idx, iwp_bins, local_time_bins, n_profiles, cell_3d, time_3d):
    iwp_mask = (ds.IWP > iwp_bins[i]) & (ds.IWP <= iwp_bins[i + 1])
    time_mask = (ds.time_local > local_time_bins[j]) & (
        ds.time_local <= local_time_bins[j + 1]
    )
    mask = iwp_mask & time_mask
    flat_mask = mask.values.flatten()
    idx_true = idx[flat_mask]
    member_idx = np.random.choice(idx_true, n_profiles, replace=False)
    member_mask = np.zeros(len(idx)) > 1
    member_mask[member_idx] = True
    cells = cell_3d[member_mask]
    times = time_3d[member_mask]
    return i, j, cells, times


def sample_profiles(
    ds,
    local_time_bins,
    local_time_points,
    iwp_bins,
    iwp_points,
    n_profiles,
    coordinates,
):
    """
    Samples profiles from the dataset based on the given bins and points.

    Args:
    - ds (xarray.Dataset): Original dataset containing the variables.
    - local_time_bins (array-like): Array of local time bins.
    - local_time_points (array-like): Array of points for local time bins.
    - iwp_bins (array-like): Array of ice water path bins.
    - iwp_points (array-like): Array of points for ice water path bins.
    - n_profiles (int): Number of profiles to sample.
    - locations (xarray.Dataset): Dataset to store the sampled locations.

    Returns:
    - None
    """

    idx = np.arange(len(ds["IWP"].values.flatten()))
    cell = ds.ncells.values
    time = ds.time.values
    cell_3d, time_3d = np.meshgrid(cell, time)
    cell_3d = cell_3d.flatten()
    time_3d = time_3d.flatten()

    # Create masks for IWP and local time bins
    iwp_masks = [
        (ds.IWP > iwp_bins[i]) & (ds.IWP <= iwp_bins[i + 1])
        for i in range(len(iwp_points))
    ]
    time_masks = [
        (ds.time_local > local_time_bins[j]) & (ds.time_local <= local_time_bins[j + 1])
        for j in range(len(local_time_points))
    ]

    for i, iwp_mask in tqdm(enumerate(iwp_masks)):
        for j, time_mask in enumerate(time_masks):
            mask = iwp_mask & time_mask
            flat_mask = mask.values.flatten()
            idx_true = idx[flat_mask]
            if len(idx_true) >= n_profiles:
                member_idx = np.random.choice(idx_true, n_profiles, replace=False)
                member_mask = np.zeros(len(idx), dtype=bool)
                member_mask[member_idx] = True
                cells = cell_3d[member_mask]
                times = time_3d[member_mask]
                coordinates.loc[{"ciwp": iwp_points[i], "ctime": local_time_points[j]}][
                    "ncells"
                ][:] = cells
                coordinates.loc[{"ciwp": iwp_points[i], "ctime": local_time_points[j]}][
                    "time"
                ][:] = times
            else:
                print(
                    f"Warning: Not enough profiles for iwp_bin {i} and time_bin {j}. Skipping."
                )

    return coordinates


# %% concatenate single files
def concatenate_files(run, followup, exp_name, file):
    datasets = {}
    path = f"/work/bm1183/m301049/icon_hcap_data/{exp_name}/production/random_sample/"
    filelist = [
        f"{path}{f}"
        for f in os.listdir(path)
        if (f.startswith(f"{run}_{file}")) or (f.startswith(f"{followup}_{file}"))
    ]
    filelist.sort()
    datasets[file] = xr.open_mfdataset(
        filelist,
        combine="nested",
        concat_dim=["index"],
    ).sortby("index")
    datasets[file].to_netcdf(f"{path}{run}_{file[:-3]}_randsample.nc")


# %% merge 2D and 3D data
def merge_files(run, filenames, exp_name):
    datasets = []
    for file in filenames:
        datasets.append(
            xr.open_dataset(
                f"/work/bm1183/m301049/icon_hcap_data/{exp_name}/production/random_sample/{run}_{file[:-3]}_randsample.nc"
            )
        )
    datasets[0] = datasets[0].rename({"height_2": "s_height_2", "height": "s_height"})
    ds = xr.merge(datasets)
    path = f"/work/bm1183/m301049/icon_hcap_data/{exp_name}/production/random_sample/{run}_randsample.nc"
    ds.to_netcdf(path)
