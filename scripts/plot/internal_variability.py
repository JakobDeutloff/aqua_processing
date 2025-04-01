# %%
import xarray as xr
import easygems.healpix as egh
import numpy as np
import matplotlib.pyplot as plt

# %%
runs = ["jed0011", "jed0033", "jed0022"]
experiments = {
    "jed0011": "control",
    "jed0033": "plus2K",
    "jed0022": "plus4K",
}
colors = {
    "jed0011": "k",
    "jed0033": "orange",
    "jed0022": "r",
}
datasets = {}
iwp = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{experiments[run]}/production/{run}_atm_2d_hp.nc"
    ).pipe(egh.attach_coords)
    iwp[run] = (
        datasets[run]["clivi"] + datasets[run]["qsvi"] + datasets[run]["qgvi"]
    ).where((datasets[run].lat < 20) & (datasets[run].lat > -20), drop=True)

# %%
iwp_stacked = {}
for run in runs:
    iwp_stacked[run] = iwp[run].stack(idx=("time", "cell")).reset_index("idx")
# %% get random samples
size = int(1e7)
number = 5
rand_samples = {}
for run in runs:
    samples = []
    for j in range(number):
        values = np.random.choice(iwp_stacked[run].idx, size=size)
        samples.append(iwp_stacked[run].sel(idx=values))
    rand_samples[run] = samples

# %% get histograms
iwp_bins = np.logspace(-4, np.log10(40), 51)
histograms = {}
diff_hists = {}
std = {}
mean = {}
for run in runs:
    hists = []
    for i in range(number):
        hist, edges = np.histogram(
            rand_samples[run][i], bins=iwp_bins, density=False
        )
        hist = hist / size
        hists.append(hist)
    histograms[run] = hists

for run in ['jed0022', 'jed0033']:
    diffs = []
    for i in range(number):
        diff = histograms[run][i] - histograms['jed0011'][i]
        diffs.append(diff)
    diff_hists[run] = diffs
    std[run] = np.std(diff_hists[run], axis=0)
    mean[run] = np.mean(diff_hists[run], axis=0)
    

# %% 
fig , ax = plt.subplots()
for run in ['jed0022', 'jed0033']:
    ax.stairs(
        mean[run],
        edges,
        label=experiments[run],
        color=colors[run],
    )
    ax.fill_between(
        edges[:-1],  # Left edges of bins
        mean[run] - std[run],  # Lower bound (mean - std)
        mean[run] + std[run],  # Upper bound (mean + std)
        step="post",  # Match the step style of stairs
        color=colors[run],
        alpha=0.3,  # Transparency for the filled area
        label=f"{experiments[run]} ± std",
    )
ax.set_xlim([1e-4, 40])
ax.set_xscale("log")
ax.set_xlabel("$I$ / kg m$^{-2}$")
ax.set_ylabel("P($I$)") 
ax.spines[['top', 'right']].set_visible(False)
ax.axhline(0, color='grey', lw=0.5, ls='--')
labels = ['+4K', '+2K', r'$\pm \sigma$']
handles = [
    plt.Line2D([0], [0], color=colors['jed0022'], lw=2),
    plt.Line2D([0], [0], color=colors['jed0033'], lw=2),
    plt.fill_between([], [], [], color='grey', alpha=0.3),
]
ax.legend(
    handles=handles,
    labels=labels,
    loc="upper right",
    fontsize=10,
)
fig.savefig('plots/variability/iwp_sampling.png', dpi=300, bbox_inches='tight')
# %%
