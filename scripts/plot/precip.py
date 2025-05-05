# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# %% load data
runs = ["jed0011", "jed0022", "jed0033"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
colors = {"jed0011": "k", "jed0022": "red", "jed0033": "orange"}
datasets = {}
cre_interp_mean = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed.nc"
    )

# %% calculate binned precip
pr_binned = {}
pr_mean = {}
pr_std = {}
lat_bins = np.arange(-90, 90.5, 0.5)
lat_points = (lat_bins[:-1] + lat_bins[1:]) / 2
for run in runs:
    pr_mean[run] = datasets[run]["pr"].groupby_bins(datasets[run]['clat'], lat_bins).mean() * 86400
    pr_std[run] = datasets[run]["pr"].groupby_bins(datasets[run]['clat'], lat_bins).std() * 86400
# %% plot binned precip
fig, ax = plt.subplots()

for run in ['jed0022', 'jed0033', 'jed0011']:
    ax.plot(lat_points, pr_mean[run], label=exp_name[run], color=colors[run])
    #ax.fill_between(lat_points, pr_mean[run] - pr_std[run], pr_mean[run] + pr_std[run], alpha=0.2, color=colors[run])

ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel("Latitude")
ax.set_ylabel("Precipitation / mm day$^{-1}$")
ax.set_xlim([-30, 30])

handles, labels = ax.get_legend_handles_labels()
handles.append(mpl.patches.Patch(color='grey', alpha=0.2))
labels.append(r'$\pm  \sigma$')
ax.legend(handles=handles, labels=labels)

fig.savefig('plots/publication/precip.png', dpi=300, bbox_inches='tight')

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
