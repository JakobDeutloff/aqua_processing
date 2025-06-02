# %% import 
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.read_data import load_daily_2d_data
import xarray as xr

# %% load data
runs = ["jed0011", "jed0022", "jed0033"]
colors = {"jed0011": "k", "jed0022": "red", "jed0033": "orange"}
line_labels = {
    "jed0011": "Control",
    "jed0022": "+4 K",
    "jed0033": "+2 K",
}
datasets = load_daily_2d_data(['clivi', 'qsvi', 'qgvi'], tropics_only=True, load=False, frequency='hour')
# %%
iwp = {}
for run in runs:
    iwp[run] = (datasets[run]['clivi'] + datasets[run]['qsvi'] + datasets[run]['qgvi']).isel(time=slice(-24, None)).load()

# %% plot individual timeseries
fig, ax = plt.subplots(figsize=(8, 4))
x = iwp['jed0011'].isel(ncells=slice(0, 10)).mean('ncells').values
a = scipy.signal.correlate(x, x, mode='full')
acf = a[len(a) // 2:] / a[len(a) // 2]
ax.plot(iwp['jed0011'].time, x, label=line_labels['jed0011'], color=colors['jed0011'])
ax.plot(iwp['jed0011'].time, acf, label='ACF', color='gray', linestyle='--')

# %% calculate autocorrelation 

a = scipy.signal.correlate(x, x, mode='full')
acf = a[len(a) // 2:] / a[len(a) // 2]
efold = np.where(acf < 1/np.e)[0][0] if np.any(acf < 1/np.e) else np.nan


# %% calculate autocorrelation

def e_folding_time(x):
    x = np.asarray(x)
    # Remove NaNs
    if np.all(np.isnan(x)):
        return np.nan
    x = x - np.nanmean(x)
    if np.allclose(x, 0) or np.isnan(x).all():
        return np.nan
    # Fill NaNs with zero (or interpolate if you prefer)
    x = np.nan_to_num(x, nan=0.0)
    acf = scipy.signal.correlate(x, x, mode='full')
    mid = len(acf) // 2
    if acf[mid] == 0:
        return np.nan
    acf = acf[mid:] / acf[mid]
    below = np.where(acf < 1/np.e)[0]
    return below[0] if len(below) > 0 else np.nan


efolding_lifetime = xr.apply_ufunc(
    e_folding_time,
    np.log10(iwp['jed0011'].values),  # shape: time x ncells
    input_core_dims=[["time"]],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[float]
)

# %%
