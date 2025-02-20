import numpy as np
import xarray as xr
from scipy.interpolate import griddata
import multiprocessing as mp
from tqdm import tqdm
from functools import partial


def calculate_hc_temperature(ds, IWP_emission=7.3e-3):
    """
    Calculate the temperature of high clouds.
    """

    top_idx_thick = np.abs(ds["iwc_cumsum"] - IWP_emission).argmin("height")
    top_idx_thin = (ds["cli"] + ds["qs"] + ds["qg"]).argmax("height")

    top_idx = xr.where(top_idx_thick < top_idx_thin, top_idx_thick, top_idx_thin)
    p_top = ds.isel(height=top_idx).phalf
    T_h = ds["ta"].isel(height=top_idx)

    T_h.attrs = {"units": "K", "long_name": "High Cloud Top Temperature"}
    p_top.attrs = {"units": "hPa", "long_name": "High cloud top pressure"}

    return T_h, p_top/100


def calc_cre(ds, mode):

    if mode == "clearsky":
        cre_net = -1 * (ds["rsut"] + ds["rlut"] - (ds["rsutcs"] + ds["rlutcs"]))
        cre_net.attrs = {
            "units": "W m^-2",
            "long_name": "Net Cloud Radiative Effect Clear Sky",
        }
        cre_sw = -1 * (ds["rsut"] - ds["rsutcs"])
        cre_sw.attrs = {
            "units": "W m^-2",
            "long_name": "Shortwave Cloud Radiative Effect Clear Sky",
        }
        cre_lw = -1 * (ds["rlut"] - ds["rlutcs"])
        cre_lw.attrs = {
            "units": "W m^-2",
            "long_name": "Longwave Cloud Radiative Effect Clear Sky",
        }

    elif mode == "wetsky":
        cre_net = -1 * (ds["rsut"] + ds["rlut"] - (ds["rsutws"] + ds["rlutws"]))
        cre_net.attrs = {
            "units": "W m^-2",
            "long_name": "Net Cloud Radiative Effect Wet Sky",
        }
        cre_sw = -1 * (ds["rsut"] - ds["rsutws"])
        cre_sw.attrs = {
            "units": "W m^-2",
            "long_name": "Shortwave Cloud Radiative Effect Wet Sky",
        }
        cre_lw = -1 * (ds["rlut"] - ds["rlutws"])
        cre_lw.attrs = {
            "units": "W m^-2",
            "long_name": "Longwave Cloud Radiative Effect Wet Sky",
        }

    else:
        raise ValueError("mode must be either clearsky or wetsky")

    return cre_net, cre_sw, cre_lw


def interpolate(ds):
    non_nan_indices = np.array(np.where(~np.isnan(ds)))
    non_nan_values = ds[~np.isnan(ds)]
    nan_indices = np.array(np.where(np.isnan(ds)))

    interpolated_values = griddata(
        non_nan_indices.T, non_nan_values, nan_indices.T, method="linear"
    )

    copy = ds.copy()
    copy[np.isnan(ds)] = interpolated_values
    return copy


def bin_and_average_cre(ds, IWP_bins, time_bins, std=False):

    dummy = np.zeros([len(IWP_bins) - 1, len(time_bins) - 1])
    cre_arr = {"net": dummy.copy(), "sw": dummy.copy(), "lw": dummy.copy()}
    if std:
        cre_arr_std = {"net": dummy.copy(), "sw": dummy.copy(), "lw": dummy.copy()}

    for i in range(len(IWP_bins) - 1):
        IWP_mask = (ds["iwp"] > IWP_bins[i]) & (ds["iwp"] < IWP_bins[i + 1])
        for j in range(len(time_bins) - 1):
            lon_mask = (ds.time_local > time_bins[j]) & (ds.time_local <= time_bins[j + 1])

            cre_arr["net"][i, j] = float(
                (ds["cre_net_hc"].where(IWP_mask & lon_mask)).mean().values
            )
            cre_arr["sw"][i, j] = float(
                (ds["cre_sw_hc"].where(IWP_mask & lon_mask)).mean().values
            )
            cre_arr["lw"][i, j] = float(
                (ds["cre_lw_hc"].where(IWP_mask & lon_mask)).mean().values
            )
            if std:
                cre_arr_std["net"][i, j] = float(
                    (ds["cre_net_hc"].where(IWP_mask & lon_mask)).std().values
                )
                cre_arr_std["sw"][i, j] = float(
                    (ds["cre_sw_hc"].where(IWP_mask & lon_mask)).std().values
                )
                cre_arr_std["lw"][i, j] = float(
                    (ds["cre_lw_hc"].where(IWP_mask & lon_mask)).std().values
                )

    # Interpolate
    interp_cre = {
        "net": cre_arr["net"].copy(),
        "sw": cre_arr["sw"].copy(),
        "lw": cre_arr["lw"].copy(),
    }

    for key in interp_cre.keys():
        interp_cre[key] = interpolate(cre_arr[key])

    # average over lat
    interp_cre_average = {}
    if std:
        cre_std_average = {}
    for key in interp_cre.keys():
        interp_cre_average[key] = np.nanmean(interp_cre[key], axis=1)
        if std:
            cre_std_average[key] = np.nanmean(cre_arr_std[key], axis=1)

    # put data into xarrays
    cre_arr = xr.Dataset(
        {
            "net": (("iwp", "time"), cre_arr["net"]),
            "sw": (("iwp", "time"), cre_arr["sw"]),
            "lw": (("iwp", "time"), cre_arr["lw"]),
        },
        coords={
            "iwp": (IWP_bins[:-1] + IWP_bins[1:]) / 2,
            "time": (time_bins[:-1] + time_bins[1:]) / 2,
        },
    )
    interp_cre = xr.Dataset(
        {
            "net": (("iwp", "time"), interp_cre["net"]),
            "sw": (("iwp", "time"), interp_cre["sw"]),
            "lw": (("iwp", "time"), interp_cre["lw"]),
        },
        coords={
            "iwp": (IWP_bins[:-1] + IWP_bins[1:]) / 2,
            "time": (time_bins[:-1] + time_bins[1:]) / 2,
        },
    )
    interp_cre_average = xr.Dataset(
        {
            "net": ("iwp", interp_cre_average["net"]),
            "sw": ("iwp", interp_cre_average["sw"]),
            "lw": ("iwp", interp_cre_average["lw"]),
        },
        coords={"iwp": (IWP_bins[:-1] + IWP_bins[1:]) / 2},
    )
    if std:
        cre_std_average = xr.Dataset(
            {
                "net": ("iwp", cre_std_average["net"]),
                "sw": ("iwp", cre_std_average["sw"]),
                "lw": ("iwp", cre_std_average["lw"]),
            },
            coords={"iwp": (IWP_bins[:-1] + IWP_bins[1:]) / 2},
        )

        return cre_arr, interp_cre, interp_cre_average, cre_std_average
    else:
        return cre_arr, interp_cre, interp_cre_average


def calc_connected(ds, frac_no_cloud=0.05):
    """
    defines for all profiles with ice above liquid whether
    the high and low clouds are connected (1) or not (0).
    Profiles where not both cloud types are present are filled with nan.
    Profiles masked aout in atm will also be nan.

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing atmospheric profiles, can be masked if needed
    frac_no_cloud : float
        Fraction of maximum cloud condensate in column to define no cloud

    Returns:
    --------
    connected : xarray.DataArray
        DataArray containing connectedness for each profile
    """

    # define liquid and ice cloud condensate
    liq = ds["clw"] + ds["qr"]
    ice = ds["cli"] + ds["qs"] + ds["qg"]

    # define ice and liquid content needed for connectedness
    no_ice_cloud = (ice > (frac_no_cloud * ice.max("height"))) * 1
    no_liq_cloud = (liq > (frac_no_cloud * liq.max("height"))) * 1
    no_cld = no_liq_cloud + no_ice_cloud

    # find all profiles with ice above liquid
    mask_both_clds = (ds["lwp"] > 1e-4) & (ds["iwp"] > 1e-4)

    # prepare coordinates with liquid clouds below ice clouds for indexing in the loop
    index_valid = ds.index[mask_both_clds]

    # create connectedness array
    connected = xr.DataArray(
        np.ones(mask_both_clds.shape),
        coords=mask_both_clds.coords,
        dims=mask_both_clds.dims,
    )
    connected.attrs = {
        "units": "1",
        "long_name": "Connectedness of liquid and frozen clouds",
    }

    # set all profiles with no liquid cloud to nan
    connected = connected.where(mask_both_clds, np.nan)

    # Calculate the height of maximum ice and liquid content
    h_ice = ice.argmax("height")
    h_liq = liq.argmax("height")

    # Calculate the mean cloud content between the heights of maximum ice and liquid content
    cld_range_mean = no_cld.where((no_cld.height >= h_ice) & (no_cld.height <= h_liq)).mean("height")

    # Determine connectedness
    connected = xr.where(cld_range_mean < 1, 0, connected)

    return connected


def calc_IWC_cumsum(ds):
    """
    Calculate the vertically integrated ice water content.
    """

    ice_mass = (ds["cli"] + ds["qg"] + ds["qs"]) * ds["dzghalf"].values
    IWC_cumsum = ice_mass.cumsum("height")

    IWC_cumsum.attrs = {
        "units": "kg m^-2",
        "long_name": "Cumulative Ice Water Content",
    }
    return IWC_cumsum
