# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.read_data import load_daily_2d_data
from scipy.ndimage import gaussian_filter1d



# %% load data
runs = ["jed0011", "jed0022", "jed0033"]
colors = {"jed0011": "k", "jed0022": "red", "jed0033": "orange"}
line_labels = {
    "jed0011": "Control",
    "jed0022": "+4 K",
    "jed0033": "+2 K",
}
datasets = load_daily_2d_data(['pr'])

# %% calculate binned precip
pr_binned = {}
pr_mean = {}
pr_std = {}
lat_bins = np.arange(-20, 20.1, 0.1)
lat_points = (lat_bins[:-1] + lat_bins[1:]) / 2
for run in runs:
    pr_mean[run] = datasets[run]["pr"].groupby_bins(datasets[run]['clat'], lat_bins).mean() * 86400
    pr_std[run] = pr_mean[run].std('time')
    pr_mean[run] = pr_mean[run].mean('time')
    
# %% smooth binned means by 
sigma = 3  
pr_mean_filt = {}
pr_std_filt = {}
for run in runs:
    pr_mean_filt[run] = gaussian_filter1d(pr_mean[run], sigma)
    pr_std_filt[run] = gaussian_filter1d(pr_std[run], sigma)
# %% plot binned precip
fig, ax = plt.subplots()

for run in ['jed0022', 'jed0033', 'jed0011']:
    ax.plot(lat_points, pr_mean_filt[run], label=line_labels[run], color=colors[run])
    ax.fill_between(lat_points, pr_mean_filt[run] - pr_std_filt[run], pr_mean_filt[run] + pr_std_filt[run], alpha=0.2, color=colors[run])

ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel("Latitude")
ax.set_ylabel("Precipitation / mm day$^{-1}$")

handles, labels = ax.get_legend_handles_labels()
handles.append(mpl.patches.Patch(color='grey', alpha=0.2))
labels.append(r'$\pm  \sigma$')
ax.legend(handles=handles, labels=labels)
ax.set_xlim([-20, 20])
ax.set_ylim([0, 20])

fig.savefig('plots/publication/precip.png', dpi=300, bbox_inches='tight')

# %% calculate hydrological sensitivity
t_deltas = {'jed0022': 4, 'jed0033': 2}
hydr_sens = {}
mean_control = datasets['jed0011']['pr'].mean().values
for run in ['jed0022', 'jed0033']:
    hydr_sens[run] = (datasets[run]['pr'].mean().values - mean_control) * 100 / mean_control / t_deltas[run]
    print(f"Hydrological sensitivity for {line_labels[run]}: {hydr_sens[run]:.2f} %/K")



# %%
