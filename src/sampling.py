import xarray as xr
import pandas as pd
from dask.diagnostics import ProgressBar
import glob
import os
from src.grid_helpers import merge_grid, fix_time
import numpy as np
from tqdm import tqdm
from dask.distributed import Client
import dask.array as da


def get_coarse_time(first, last, path):
    """
    Build new time axis with 6h frequency
    """
    ds_first = xr.open_dataset(path + first, chunks={})
    ds_last = xr.open_dataset(path + last, chunks={})
    time = pd.date_range(
        start=ds_first.isel(time=1).time.values,
        end=ds_last.isel(time=-1).time.values,
        freq="6h",
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
        ds_coarse = ds.sel(time=time_subset)
        with ProgressBar():
            ds_coarse.to_netcdf(file[:-3] + "_coarse.nc")
            os.remove(file)
            os.rename(file[:-3] + "_coarse.nc", file)


def subsample_file(files, exp_name, number):
    """
    Get random sample of data from file
    """
    ds = (
        xr.open_mfdataset(
            files,
            chunks="auto" ,
        )
        .pipe(merge_grid)
        .pipe(fix_time)
    )
    random_coords = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name}/production/random_sample/random_coords{number}.nc"
    )
    # coarsen to 6h timestep
    time_coarse = pd.date_range(start=ds.time.values[0], end=ds.time.values[-1], freq='6h')
    ds = ds.sel(time=time_coarse)

    # select data
    print("Selecting data")
    ds_random = ds.sel(
        ncells=random_coords.ncells.astype(int), time=random_coords.time
    ).assign_coords(
        time=random_coords.time,
        ncells=random_coords.ncells,
        clat=ds.clat.sel(ncells=random_coords.ncells),
        clon=ds.clon.sel(ncells=random_coords.ncells),
    )

    # save in random sample folder
    filename = files.split("/")[-1]
    print("Saving data")
    with ProgressBar():
        ds_random.to_netcdf(
            f"/work/bm1183/m301049/icon_hcap_data/{exp_name}/production/random_sample/{filename[:-2]}_rand{number}.nc"
        )

def get_random_coords(run, model_config, exp_name, number=0):

    path = f"/work/bm1183/m301049/{model_config}/experiments/{run}/"
    ds_3D = (
        xr.open_mfdataset(f"{path}{run}_atm_3d_main_19*.nc", chunks={})
        .pipe(merge_grid)
        .pipe(fix_time)
    )

    # select tropics
    ds_3D_trop = ds_3D.where((ds_3D.clat < 30) & (ds_3D.clat > -30), drop=True)

    # get random coordinates across time and ncells
    ncells = ds_3D_trop.sizes["ncells"]
    time = ds_3D_trop.sizes["time"]

    # Generate unique pairs of random indices
    num_samples = int(1e7)
    total_indices = ncells * time
    random_indices = da.random.randint(0, total_indices, num_samples).compute()

    random_ncells_idx = random_indices % ncells
    random_time_idx = random_indices // ncells

    # create xrrays
    random_coords = xr.Dataset(
        {
            "time": xr.DataArray(ds_3D_trop.time[random_time_idx].values, dims="index"),
            "ncells": xr.DataArray(
                ds_3D_trop.ncells[random_ncells_idx].values, dims="index"
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


def sample_profiles_old(
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
    # sample profiles
    idx = np.arange(len(ds["IWP"].values.flatten()))
    cell = ds.ncells.values
    time = ds.time.values
    cell_3d, time_3d = np.meshgrid(cell, time)
    cell_3d = cell_3d.flatten()
    time_3d = time_3d.flatten()

    for i in tqdm(range(len(iwp_points))):
        iwp_mask = (ds.IWP > iwp_bins[i]) & (ds.IWP <= iwp_bins[i + 1])
        for j in range(len(local_time_points)):
            time_mask = (ds.time_local > local_time_bins[j]) & (
                ds.time_local <= local_time_bins[j + 1]
            )
            mask = iwp_mask & time_mask
            # select n_profiles random true members from mask and set the rest to false
            flat_mask = mask.values.flatten()
            idx_true = idx[flat_mask]
            member_idx = np.random.choice(idx_true, n_profiles, replace=False)
            member_mask = np.zeros(len(idx)) > 1
            member_mask[member_idx] = True
            cells = cell_3d[member_mask]
            times = time_3d[member_mask]
            coordinates.loc[{"ciwp": iwp_points[i], "ctime": local_time_points[j]}][
                "ncells"
            ][:] = cells
            coordinates.loc[{"ciwp": iwp_points[i], "ctime": local_time_points[j]}][
                "time"
            ][:] = times

    return coordinates
