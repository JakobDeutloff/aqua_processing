import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from tqdm import tqdm


def calculate_IWC_cumsum(atms, convention="icon"):
    """
    Calculate the vertically integrated ice water content.
    """
    cell_height = 0  # needs to be extracted from vgrid
    # pressure coordinate needs to be reversed for cumsum
    ice_mass = ((atms["IWC"] + atms["graupel"] + atms["snow"]) * cell_height).reindex(
        pressure=list(reversed(atms.pressure))
    )
    IWC_cumsum = ice_mass.cumsum("pressure").reindex(
        pressure=list(reversed(atms.pressure))
    )

    IWC_cumsum.attrs = {
        "units": "kg m^-2",
        "long_name": "Cumulative Ice Water Content",
    }
    return IWC_cumsum


def calculate_h_cloud_temperature(
    atms, fluxes, IWP_emission=8e-3, convention="icon", option="bright"
):
    """
    Calculate the temperature of high clouds.
    """
    if convention == "icon":
        vert_coord = "level_full"
    elif convention == "arts":
        vert_coord = "pressure"

    if option == "bright":
        # calc brightness temperature
        flux = np.abs(fluxes["allsky_lw_up"].isel(pressure=-1))
        T_bright = (flux / 5.67e-8) ** (1 / 4)
        # exclude temperatures above tropopause
        p_trop_ixd = atms["temperature"].argmin(vert_coord)
        p_trop = atms["pressure"].isel(level_full=p_trop_ixd)
        T_profile = atms["temperature"].where(atms["pressure"] > p_trop, 0)
        # find pressure level where T == T_bright in troposphere
        top_idx_thick = np.abs(T_profile - T_bright).argmin(vert_coord)
    elif option == "emission":
        top_idx_thick = np.abs(atms["IWC_cumsum"] - IWP_emission).argmin(vert_coord)

    top_idx_thin = (atms["IWC"] + atms["snow"] + atms["graupel"]).argmax(vert_coord)

    if convention == "icon":
        top_idx = xr.where(top_idx_thick < top_idx_thin, top_idx_thick, top_idx_thin)
        p_top = atms.isel(level_full=top_idx).pressure
        T_h = atms["temperature"].isel(level_full=top_idx)
    elif convention == "arts":
        top_idx = xr.where(top_idx_thick > top_idx_thin, top_idx_thick, top_idx_thin)
        p_top = atms.isel(pressure=top_idx).pressure
        T_h = atms["temperature"].sel(pressure=p_top)
    T_h.attrs = {"units": "K", "long_name": "High Cloud Top Temperature"}
    p_top.attrs = {"units": "hPa", "long_name": "High cloud top pressure"}

    return T_h, p_top


def calc_cre(ds, mode):

    cre = xr.Dataset(coords=ds.coords)

    if mode == "clearsky":
        cre["net"] = ds["rsut"] + ds["rlut"] - (ds["rsutcs"] + ds["rlutcs"])
        cre["sw"] = ds["rsut"] - ds["rsutcs"]
        cre["lw"] = ds["rlut"] - ds["rlutcs"]

    elif mode == "noice":
        cre["net"] = ds["rsut"] + ds["rlut"] - (ds["rsutws"] + ds["rlutws"])
        cre["sw"] = ds["rsut"] - ds["rsutws"]
        cre["lw"] = ds["rlut"] - ds["rlutws"]

    else:
        raise ValueError("mode must be either clearsky or noice")

    return cre


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


def bin_and_average_cre(cre, IWP_bins, time_bins, atms):

    dummy = np.zeros([len(IWP_bins) - 1, len(time_bins) - 1])
    cre_arr = {"net": dummy.copy(), "sw": dummy.copy(), "lw": dummy.copy()}
    cre_arr_std = {"net": dummy.copy(), "sw": dummy.copy(), "lw": dummy.copy()}

    for i in range(len(IWP_bins) - 1):
        IWP_mask = (atms["IWP"] > IWP_bins[i]) & (atms["IWP"] < IWP_bins[i + 1])
        for j in range(len(time_bins) - 1):
            lon_mask = (atms.lon > time_bins[j]) & (atms.lon <= time_bins[j + 1])

            cre_arr["net"][i, j] = float(
                (cre["net"].where(IWP_mask & lon_mask)).mean().values
            )
            cre_arr["sw"][i, j] = float(
                (cre["sw"].where(IWP_mask & lon_mask)).mean().values
            )
            cre_arr["lw"][i, j] = float(
                (cre["lw"].where(IWP_mask & lon_mask)).mean().values
            )
            cre_arr_std["net"][i, j] = float(
                (cre["net"].where(IWP_mask & lon_mask)).std().values
            )
            cre_arr_std["sw"][i, j] = float(
                (cre["sw"].where(IWP_mask & lon_mask)).std().values
            )
            cre_arr_std["lw"][i, j] = float(
                (cre["lw"].where(IWP_mask & lon_mask)).std().values
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
    cre_std_average = {}
    for key in interp_cre.keys():
        interp_cre_average[key] = np.nanmean(interp_cre[key], axis=1)
        cre_std_average[key] = np.nanmean(cre_arr_std[key], axis=1)

    return cre_arr, interp_cre, interp_cre_average, cre_std_average


def calc_connected(atms, frac_no_cloud=0.05):
    """
    defines for all profiles with ice above liquid whether
    the high and low clouds are connected (1) or not (0).
    Profiles where not both cloud types are present are filled with nan.
    Profiles masked aout in atm will also be nan.

    Parameters:
    -----------
    atms : xarray.Dataset
        Dataset containing atmospheric profiles, can be masked if needed
    frac_no_cloud : float
        Fraction of maximum cloud condensate in column to define no cloud

    Returns:
    --------
    connected : xarray.DataArray
        DataArray containing connectedness for each profile
    """

    # define liquid and ice cloud condensate
    liq = atms["clw"] + atms["qr"]
    ice = atms["cli"] + atms["qs"] + atms["qg"]

    # define ice and liquid content needed for connectedness
    no_ice_cloud = (ice > (frac_no_cloud * ice.max("height"))) * 1
    no_liq_cloud = (liq > (frac_no_cloud * liq.max("height"))) * 1
    no_cld = no_liq_cloud + no_ice_cloud

    # find all profiles with ice above liquid
    mask_liquid_clds = (atms["lwp"] > 1e-4)

    # prepare coordinates with liquid clouds below ice clouds for indexing in the loop
    n_profiles = int((mask_liquid_clds * 1).sum().values)
    index_valid = atms.index[mask_liquid_clds]

    # create connectedness array
    connected = xr.DataArray(
        np.ones(mask_liquid_clds.shape),
        coords=mask_liquid_clds.coords,
        dims=mask_liquid_clds.dims,
    )
    connected.attrs = {
        "units": "1",
        "long_name": "Connectedness of liquid and frozen clouds",
    }

    # set all profiles with no liquid cloud to nan
    connected = connected.where(mask_liquid_clds)

    # loop over all profiles with ice above liquid and check for connectedness
    for i in tqdm(range(n_profiles)):
        liq_point = liq.sel(index=index_valid[i])
        ice_point = ice.sel(index=index_valid[i])
        p_top_idx = ice_point.argmax("height").values
        p_bot_idx = liq_point.argmax("height").values
        cld_range = no_cld.sel(index=index_valid[i]).isel(
            height=slice(p_bot_idx, p_top_idx)
        )
        # high and low clouds are not connected if there is a 1-cell deep layer without cloud
        for j in range(len(cld_range.height)):
            if cld_range.isel(height=j).sum() == 0:
                connected.loc[{"index": index_valid[i]}] = 0
                break

    return connected
