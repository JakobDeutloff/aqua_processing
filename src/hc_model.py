"""
Source code of the conceptual model
"""
# %% import
import numpy as np
import pandas as pd
import xarray as xr

# define functions to calculate model quantities

def logistic(x, L, x0, k, j):
    """
    Logistic function.

    PARAMETERS:
    ---------------------------
    x: array-like
        Input data.
    L: float
        Maximum value.
    x0: float
        Midpoint.
    k: float
        Steepness.
    j: float
        Offset.

    RETURNS:
    ---------------------------
    y: array-like
        Output data.
    """
    return L / (1 + np.exp(-k * (x - x0))) + j


def calc_lc_fraction(LWP, connected, threshold=1e-4):
    """
    Calculates the low cloud fraction.

    PARAMETERS:
    ---------------------------
    LWP: array-like
        Liquid water path.
    threshold: float
        Threshold value for low cloud fraction.
    connected: array-like
        Connectedness data.

    RETURNS:
    ---------------------------
    lc_fraction: array-like
        Low cloud fraction.
    """
    lc_fraction = ((LWP >= threshold) & (connected != 1)) * 1
    return lc_fraction


def binning(IWP_bins, data, IWP):
    """
    Bins the data based on IWP.

    PARAMETERS:
    ---------------------------
    IWP_bins: array-like
        Bins for IWP.
    data: array-like
        Data to be binned.
    IWP: array-like
        Corresponding IWP.

    RETURNS:
    ---------------------------
    binned_data: array-like
        Binned data.
    """
    if type(data) == float or type(data) == np.float64:
        return data
    else:
        return data.groupby_bins(IWP, IWP_bins).mean()


def calc_hc_albedo(IWP, alpha_hc_params):
    """
    Calculates the high-cloud albedo.

    PARAMETERS:
    ---------------------------
    IWP: array-like
        Input data.
    alpha_hc_params: tuple
        Parameters for the logistic function.

    RETURNS:
    ---------------------------
    fitted_vals: array-like
        High cloud albedo.
    """
    fitted_vals = logistic(np.log10(IWP), *alpha_hc_params, 0)
    return fitted_vals


def calc_hc_emissivity(IWP, em_hc_params, prescribed_hc_em=None):
    """
    Calculates the high-cloud emissivity.

    PARAMETERS:
    ---------------------------
    IWP: array-like
        Input data.
    em_hc_params: tuple
        Parameters for the logistic function.

    RETURNS:
    ---------------------------
    fitted_vals: array-like
        High cloud emissivity.
    """
    
    if prescribed_hc_em is not None:
        fitted_vals = prescribed_hc_em.values.squeeze()
    else:
        fitted_vals = logistic(np.log10(IWP), *em_hc_params, 0)
    return fitted_vals


def calc_alpha_t(
    lc_fraction,
    albedo_cs,
    albedo_l,
    prescribed_lc_quantities,
):
    """
    Calculates the albedo below the high clouds.

    PARAMETERS:
    ---------------------------
    LWP: array-like
        Liquid water path.
    lc_fraction: array-like
        Low cloud fraction.
    albedo_cs: float
        Clearsky albedo.
    alpha_t_params: tuple
        Parameters for the logistic function.
    const_lc_quantities: dict, optional
        Constant values for lc quantities if they should be used instead of logistic function.
    prescribed_lc_quantities: dict, optional
        Prescribed values for lc quantities.

    RETURNS:
    ---------------------------
    avg_value: array-like
        Albedo below high clouds.
    """
    avg_value = lc_fraction * albedo_l + (1 - lc_fraction) * albedo_cs
    if prescribed_lc_quantities is not None:
        avg_value = prescribed_lc_quantities["a_t"].values
    return avg_value


def calc_R_t(
    IWP,
    lc_fraction,
    R_cs,
    R_l,
    c_h20_params,
    prescribed_lc_quantities,
):
    """
    Calculates the LW radiation from below the high clouds.

    PARAMETERS:
    ---------------------------
    LWP: array-like
        Liquid water path.
    IWP: array-like
        Ice water path.
    lc_fraction: array-like
        Low cloud fraction.
    R_t_cs: float
        Clearsky R_t.
    R_t_params: tuple
        Parameters for the linear regression.
    h20_params: tuple
        Parameters for the linear regression.
    const_lc_quantities: dict, optional
        Constant values for lc quantities if they should be used instead of linear regression.
    prescribed_lc_quantities: dict, optional
        Prescribed values for lc quantities.

    RETURNS:
    ---------------------------
    avg_value: array-like
        LW radiation from below high clouds.
    """
    h2o_correction = c_h20_params.slope * np.log10(IWP) + c_h20_params.intercept
    avg_value = lc_fraction * R_l + (1 - lc_fraction) * R_cs + h2o_correction
    if prescribed_lc_quantities is not None:
        avg_value = prescribed_lc_quantities["R_t"].values
    return avg_value


def hc_sw_cre(alpha_hc, alpha_t, SW_in):
    """
    Calculates the SW CRE of the high clouds.

    PARAMETERS:
    ---------------------------
    alpha_hc: array-like
        High cloud albedo.
    alpha_t: array-like
        Albedo below high clouds.
    SW_in: float
        Daily average SW radiation at TOA.

    RETURNS:
    ---------------------------
    SW_cre: array-like
        SW CRE of the high clouds.
    """
    return -SW_in * (
        alpha_hc + (alpha_t * (1 - alpha_hc) ** 2) / (1 - alpha_t * alpha_hc) - alpha_t
    )


def hc_lw_cre(em_hc, T_h, R_t, sigma):
    """
    Calculates the LW CRE of the high clouds.

    PARAMETERS:
    ---------------------------
    em_hc: array-like
        High cloud emissivity.
    T_h: array-like
        High cloud temperature.
    R_t: array-like
        LW radiation from below high clouds.
    sigma: float
        Stefan-Boltzmann constant.

    RETURNS:
    ---------------------------
    LW_cre: array-like
        LW CRE of the high clouds.
    """
    return em_hc * ((-1 * sigma * T_h**4) - R_t)


def run_model(
    IWP_bins,
    T_hc,
    LWP,
    IWP,
    connectedness,
    parameters,
    prescribed_lc_quantities=None,
    prescribed_hc_em=None,
):
    """
    Runs the HC Model with given input data and parameters.

    PARAMETERS:
    ---------------------------
    IWP_bins: array-like
        Bins for IWP.
    T_hc: array-like
        High cloud temperature.
    LWP: array-like
        Liquid water path.
    IWP: array-like
        Ice water path.
    connectedness: array-like
        Connectedness data.
    parameters: dict
        Parameters for the model.
    prescribed_lc_quantities: dict, optional
        Prescribed values for lc quantities

    RETURNS:
    ---------------------------
    results: pd.DataFrame
        Results of the model.
    """

    IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2

    # calculate lc fraction if needed
    if parameters["f"] is None:
        lc_fraction = calc_lc_fraction(LWP, connected=connectedness)
    else:
        lc_fraction = parameters["f"]

    # bin the input data
    T_hc_binned = binning(IWP_bins, T_hc, IWP)
    LWP_binned = binning(IWP_bins, LWP.where(LWP > 1e-4), IWP)
    lc_fraction_binned = binning(IWP_bins, lc_fraction, IWP)

    # calculate radiative properties below high clouds
    alpha_t = calc_alpha_t(
        lc_fraction=lc_fraction_binned,
        albedo_cs=parameters['a_cs'],
        albedo_l=parameters["a_l"],
        prescribed_lc_quantities=prescribed_lc_quantities,
    )
    R_t = calc_R_t(
        IWP_points,
        lc_fraction_binned,
        parameters["R_cs"],
        parameters["R_l"],
        parameters["c_h2o"],
        prescribed_lc_quantities,
    )

    # calculate radiative properties of high clouds
    alpha_hc = calc_hc_albedo(IWP_points, parameters["alpha_hc"])
    em_hc = calc_hc_emissivity(IWP_points, parameters["em_hc"], prescribed_hc_em)

    # calculate HCRE
    SW_cre = hc_sw_cre(alpha_hc, alpha_t, parameters["SW_in"])
    LW_cre = hc_lw_cre(em_hc, T_hc_binned, R_t, sigma=5.67e-8)

    # build results df
    results = pd.DataFrame()
    results.index = IWP_points
    results.index.name = "IWP"
    results["T_hc"] = T_hc_binned
    results["LWP"] = LWP_binned
    results["lc_fraction"] = lc_fraction_binned
    results["alpha_t"] = alpha_t
    results["R_t"] = R_t
    results["alpha_hc"] = alpha_hc
    results["em_hc"] = em_hc
    results["SW_cre"] = SW_cre
    results["LW_cre"] = LW_cre

    return results
