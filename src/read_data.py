import xarray as xr
import pickle
from src.grid_helpers import merge_grid, fix_time

runs = ["jed0011", "jed0022", "jed0033"]
followups = {"jed0011": "jed0111", "jed0022": "jed0222", "jed0033": "jed0333"}
configs = {"jed0011": "icon-mpim", "jed0022": "icon-mpim-4K", "jed0033": "icon-mpim-2K"}
experiments = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}


def read_cloudsat(year):
    """
    Function to read CloudSat for a given year
    """

    path_cloudsat = "/work/bm1183/m301049/cloudsat/"
    cloudsat = xr.open_dataset(
        path_cloudsat + year + "-07-01_" + str(int(year) + 1) + "-07-01_fwp.nc"
    )
    # convert ot pandas
    cloudsat = cloudsat.to_pandas()
    # select tropics
    lat_mask = (cloudsat["lat"] <= 20) & (cloudsat["lat"] >= -20)

    return cloudsat[lat_mask]


def load_parameters(run):
    """
    Load the parameters needed for the model.

    Returns
    -------
    dict
        Dictionary containing the parameters.
    """

    with open(f"data/params/{run}_hc_albedo_params.pkl", "rb") as f:
        hc_albedo = pickle.load(f)
    with open(f"data/params/{run}_hc_emissivity_params.pkl", "rb") as f:
        hc_emissivity = pickle.load(f)
    with open(f"data/params/{run}_C_h2o_params.pkl", "rb") as f:
        c_h2o = pickle.load(f)
    with open(f"data/params/{run}_lower_trop_params.pkl", "rb") as f:
        lower_trop_params = pickle.load(f)

    return {
        "alpha_hc": hc_albedo,
        "em_hc": hc_emissivity,
        "c_h2o": c_h2o,
        "R_l": lower_trop_params["R_l"],
        "R_cs": lower_trop_params["R_cs"],
        "f": lower_trop_params["f"],
        "a_l": lower_trop_params["a_l"],
        "a_cs": lower_trop_params["a_cs"],
    }


def load_lt_quantities(run):
    """
    Load the lower tropospheric quantities needed for the model.

    Returns
    -------
    dict
        Dictionary containing the lower tropospheric quantities.
    """

    with open(f"data/{run}_lower_trop_vars_mean.pkl", "rb") as f:
        lt_quantities = pickle.load(f)

    return lt_quantities


def load_iwp_hists():
    """
    Load the IWP histograms for the model.

    Returns
    -------
    dict
        Dictionary containing the IWP histograms.
    """
    iwp_hists = {}
    for run in runs:
        with open(
            f"/work/bm1183/m301049/icon_hcap_data/{experiments[run]}/production/{run}_iwp_hist.pkl",
            "rb",
        ) as f:
            iwp_hists[run] = pickle.load(f)

    return iwp_hists


def load_random_datasets(version="processed"):
    """
    Load the random datasets for the model.

    Parameters
    ----------
    processed : bool, optional
        If True, load the processed datasets, otherwise load the raw datasets. Default is True.

    Returns
    -------
    dict
        Dictionary containing the random datasets.
    """

    datasets = {}
    if version == "processed":
        for run in runs:
            datasets[run] = xr.open_dataset(
                f"/work/bm1183/m301049/icon_hcap_data/{experiments[run]}/production/random_sample/{run}_randsample_processed_64.nc"
            )
    elif version == "temp":
        for run in runs:
            datasets[run] = xr.open_dataset(
                f"/work/bm1183/m301049/icon_hcap_data/{experiments[run]}/production/random_sample/{run}_randsample_tgrid_20.nc"
            )
    else:
        for run in runs:
            datasets[run] = xr.open_dataset(
                f"/work/bm1183/m301049/icon_hcap_data/{experiments[run]}/production/random_sample/{run}_randsample.nc"
            )
    return datasets


def load_vgrid():
    """
    Load the vertical grid for the model.

    Returns
    -------
    xarray.Dataset
        Dataset containing the vertical grid.
    """

    vgrid = (
        xr.open_dataset(
            "/work/bm1183/m301049/icon-mpim/experiments/jed0001/atm_vgrid_angel.nc"
        )
        .mean("ncells")
        .rename({"height": "height_2", "height_2": "height"})
    )

    return vgrid


def load_cre():
    """
    Load  HCRE.

    Returns
    -------
    dict
        Dictionary containing the CRE data.
    """

    cre_data = {}
    for run in runs:
        cre_data[run] = xr.open_dataset(
            f"/work/bm1183/m301049/icon_hcap_data/{experiments[run]}/production/cre/{run}_cre_raw.nc"
        )

    return cre_data


def load_daily_2d_data(vars, frequency="day", tropics_only=True, load=True):
    """
    Load the one timestep per day 2D data.

    Parameters
    ----------
    vars : list
        List of variables to load.
    frequency : str, optional
        Frequency of the data, either 'day' or 'hour'. Default is 'day'.
    tropics_only : bool, optional
        If True, only load data for the tropics (latitude between -20 and 20). Default is True.
    load : bool, optional
        If True, load the data into memory. Default is True.

    Returns
    -------
    dict
        Dictionary containing the 2D data.
    """
    datasets = {}
    for run in runs:
        ds_first_month = (
            xr.open_mfdataset(
                f"/work/bm1183/m301049/{configs[run]}/experiments/{run}/{run}_atm_2d_19*.nc"
            )
            .pipe(merge_grid)
            .pipe(fix_time)[vars]
        )
        ds_last_two_months = (
            xr.open_mfdataset(
                f"/work/bm1183/m301049/{configs[run]}/experiments/{followups[run]}/{followups[run]}_atm_2d_19*.nc"
            )
            .pipe(merge_grid)
            .pipe(fix_time)[vars]
        )
        if frequency == "day":
            ds_first_month = ds_first_month.sel(
                time=(ds_first_month.time.dt.minute == 0)
                & (ds_first_month.time.dt.hour == 0)
            )
            ds_last_two_months = ds_last_two_months.sel(
                time=(ds_last_two_months.time.dt.minute == 0)
                & (ds_last_two_months.time.dt.hour == 0)
            )
        elif frequency == "hour":
            ds_first_month = ds_first_month.sel(
                time=(ds_first_month.time.dt.minute == 0)
            )
            ds_last_two_months = ds_last_two_months.sel(
                time=(ds_last_two_months.time.dt.minute == 0)
            )

        if tropics_only:
            ds_first_month = ds_first_month.where(
                (ds_first_month["clat"] < 20) & (ds_first_month["clat"] > -20),
                drop=True,
            )
            ds_last_two_months = ds_last_two_months.where(
                (ds_last_two_months["clat"] < 20) & (ds_last_two_months["clat"] > -20),
                drop=True,
            )

        if load:
            datasets[run] = xr.concat(
                [ds_first_month, ds_last_two_months], dim="time"
            ).load()
        else:
            datasets[run] = xr.concat([ds_first_month, ds_last_two_months], dim="time")
    return datasets


def load_daily_average_2d_data(vars):
    """
    Load the daily average 2D data.

    Parameters
    ----------
    vars : list
        List of variables to load.

    Returns
    -------
    dict
        Dictionary containing the daily average 2D data.
    """
    datasets = {}
    for run in runs:
        ds_first_month = (
            xr.open_mfdataset(
                f"/work/bm1183/m301049/{configs[run]}/experiments/{run}/{run}_atm_2d_daymean_19*.nc"
            )
            .pipe(merge_grid)
            .pipe(fix_time)[vars]
        )
        ds_last_two_months = (
            xr.open_mfdataset(
                f"/work/bm1183/m301049/{configs[run]}/experiments/{followups[run]}/{followups[run]}_atm_2d_daymean_19*.nc"
            )
            .pipe(merge_grid)
            .pipe(fix_time)[vars]
        )
        datasets[run] = xr.concat(
            [ds_first_month, ds_last_two_months], dim="time"
        ).load()

    return datasets


def load_cape_cin():
    """
    Load the CAPE and CIN data.

    Returns
    -------
    dict
        Dictionary containing the CAPE and CIN data.
    """
    cape_cin_data = {}
    for run in runs:
        cape_cin_data[run] = xr.open_dataset(
            f"/work/bm1183/m301049/icon_hcap_data/{experiments[run]}/production/random_sample/{run}_cape_cin.nc"
        )

    return cape_cin_data


def load_definitions():
    """
    Load the definitions for the runs, experiment names, colors, and labels.
    Returns
    -------
    tuple
        A tuple containing the runs, experiment names, colors, and labels.
    """

    runs = ["jed0011", "jed0033", "jed0022"]
    exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
    colors = {"jed0011": "#462d7b", "jed0022": "#c1df24", "jed0033": "#1f948a"}
    linestyles = {
        "jed0011": "-.",
        "jed0022": "-",
        "jed0033": "--",
    }
    sw_color = "#125fd3"
    lw_color = "#bd154a"
    net_color = "#000000"
    labels = {
        "jed0011": "Control",
        "jed0022": "+4 K",
        "jed0033": "+2 K",
    }
    return runs, exp_name, colors, labels, sw_color, lw_color, net_color, linestyles
