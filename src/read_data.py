import xarray as xr
import pickle

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