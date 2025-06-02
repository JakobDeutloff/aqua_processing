import xarray as xr
import numpy as np
from src.read_data import load_random_datasets

# %% load data
runs = ["jed0011", "jed0022", "jed0033"]
datasets = load_random_datasets(processed=True)

# %% calculate masks
mode = "raw"
mask_type = "raw"
masks_height = {}

if mask_type == "raw":
    for run in runs:
        masks_height[run] = True
elif mask_type == "simple_filter":
    for run in runs:
        masks_height[run] = datasets[run]["hc_top_temperature"] < (273.15 - 35)
elif mask_type == "dist_filter":
    masks_height = {}
    iwp_bins = np.logspace(-4, np.log10(40), 51)
    masks_height["jed0011"] = datasets["jed0011"]["hc_top_temperature"] < datasets[
        "jed0011"
    ]["hc_top_temperature"].where(datasets["jed0011"]["iwp"] > 1e-4).quantile(0.90)
    quantiles = (
        (masks_height["jed0011"] * 1)
        .groupby_bins(datasets["jed0011"]["iwp"], iwp_bins)
        .mean()
    )

    for run in runs[1:]:
        mask = xr.DataArray(
            np.ones_like(datasets[run]["hc_top_temperature"]),
            dims=datasets[run]["hc_top_temperature"].dims,
            coords=datasets[run]["hc_top_temperature"].coords,
        )
        for i in range(len(iwp_bins) - 1):
            mask_ds = (datasets[run]["iwp"] > iwp_bins[i]) & (
                datasets[run]["iwp"] <= iwp_bins[i + 1]
            )
            temp_vals = datasets[run]["hc_top_temperature"].where(mask_ds)
            mask_temp = temp_vals > temp_vals.quantile(quantiles[i])
            # mask n_masked values with the highest temperatures from temp_vals
            mask = xr.where(mask_ds & mask_temp, 0, mask)
        masks_height[run] = mask