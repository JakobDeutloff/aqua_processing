# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from src.read_data import load_random_datasets
from src.calc_variables import calc_heating_rates

# %% load data
runs = ["jed0011", "jed0033", "jed0022"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
datasets = load_random_datasets()


vgrid = (
    xr.open_dataset(
        "/work/bm1183/m301049/icon-mpim/experiments/jed0001/atm_vgrid_angel.nc"
    )
    .mean("ncells")
    .rename({"height": "height_2", "height_2": "height"})
)

# %% get moist and dry profiles
hrs_dry = {}
hrs_moist = {}
for run in runs:
    print(run)
    mask_moist = datasets[run]["iwp"] > 1
    mask_dry = datasets[run]["iwp"] < 1e-10
    hrs_dry[run] = calc_heating_rates(
        datasets[run]['rho'].where(mask_dry),
        datasets[run]['rsd'].where(mask_dry) - datasets[run]['rsu'].where(mask_dry),
        datasets[run]['rld'].where(mask_dry) - datasets[run]['rlu'].where(mask_dry),
        vgrid['zg'],
        'height'
    )
    hrs_moist[run] = calc_heating_rates(
        datasets[run]['rho'].where(mask_moist),
        datasets[run]['rsd'].where(mask_moist) - datasets[run]['rsu'].where(mask_moist),
        datasets[run]['rld'].where(mask_moist) - datasets[run]['rlu'].where(mask_moist),
        vgrid['zg'],
        'height'
    )

# %% bin heating rates by local time and calculate mean temperature profile
hrs_dry_binned = {}
hrs_moist_binned = {}
t_profiles_dry = {}
t_profiles_moist = {}
time_bins = np.arange(0, 25, 1)
time_points = (time_bins[:-1] + time_bins[1:]) / 2
for run in runs:
    print(run)
    hrs_dry_binned[run] = (
        hrs_dry[run]["net_hr"]
        .groupby_bins(datasets[run]["time_local"], bins=time_bins)
        .mean()
    )
    hrs_moist_binned[run] = (
        hrs_moist[run]["net_hr"]
        .groupby_bins(datasets[run]["time_local"], bins=time_bins)
        .mean()
    )
    t_profiles_dry[run] = (
        datasets[run]["ta"]
        .where(datasets[run]["iwp"] < 1e-10)
        .mean('index')
    )
    t_profiles_moist[run] = (
        datasets[run]["tg"]
        .where(datasets[run]["iwp"] > 1)
        .mean('index')
    )

# %% plot 
fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharey=True, sharex=True)

# plot control run  in first row
axes[0, 0].pcolormesh(
    time_points,
    t_profiles_dry['jed0011'], 
    hrs_dry_binned['jed0011'].T,

)
