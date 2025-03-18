# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from src.grid_helpers import merge_grid

# %%
runs = ["jed0011", "jed0022", "jed0033"]
build_names = {'jed0011': 'icon-mpim', 'jed0022': 'icon-mpim-4K', 'jed0033': 'icon-mpim-2K'}
exp_name = {"jed0011": "Control", "jed0022": "4K", "jed0033": "2K"}
colors = {"jed0011": "k", "jed0022": "r", "jed0033": "orange"}
datasets = {}
for run in runs:
    datasets[run] = xr.open_mfdataset(
        f"/work/bm1183/m301049/{build_names[run]}/experiments/{run}/{run}_atm_2d_daymean*.nc"
    ).pipe(merge_grid)

# %% calculate binned precip
pr_binned = {}
lat_bins = np.arange(-90, 90.5, 0.5)
lat_points = (lat_bins[:-1] + lat_bins[1:]) / 2
for run in runs:
    pr_binned[run] = datasets[run]["pr"].mean('time').groupby_bins(datasets[run]['clat'], lat_bins).mean()

# %% plot binned precip
fig, ax = plt.subplots()

for run in runs:
    ax.plot(lat_points, pr_binned[run], label=exp_name[run], color=colors[run])

ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel("Latitude")
ax.set_ylabel("Precipitation / mm/day")
ax.legend()
ax.set_xlim([-30, 30])
fig.savefig('plots/precip.png', dpi=300, bbox_inches='tight')

# %% calculate hydrological sensitivity
t_deltas = {'jed0022': 4, 'jed0033': 2}
pr_eff = {}
mean_control = datasets['jed0011']['pr'].mean().values
for run in ['jed0022', 'jed0033']:
    pr_eff[run] = (datasets[run]['pr'].mean().values - mean_control) * 100 / mean_control / t_deltas[run]
    print(f"Hydrological sensitivity for {exp_name[run]}: {pr_eff[run]:.2f} %/K")

# %% calculate energy balance 
r_surf = {}
r_toa = {}
sh = {}
p = {}
for run in runs: 
    r_surf[run] = datasets[run]['rsus'].mean().values - datasets[run]['rsds'].mean().values + datasets[run]['rlus'].mean().values - datasets[run]['rlds'].mean().values 
    r_toa[run] = datasets[run]['rsdt'].mean().values - datasets[run]['rsut'].mean().values - datasets[run]['rlut'].mean().values
    sh[run] = datasets[run]['hfss'].mean().values
    p[run] = datasets[run]['pr'].mean().values


# %%
atm_cooling = {}
cooling_increase = {}
energy_p = {}
L = 2.45e6  # latent heat of vaporization in J/kg
for run in runs:
    atm_cooling[run] = r_toa[run] + r_surf[run] + sh[run]
    energy_p[run] = np.abs(atm_cooling[run]) / L
    cooling_increase[run] = (atm_cooling[run] - atm_cooling['jed0011']) * 100 / atm_cooling['jed0011']




# %%
