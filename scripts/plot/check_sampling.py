# %%
import xarray as xr
import easygems.healpix as egh
import numpy as np
import matplotlib.pyplot as plt

# %%
ds = xr.open_dataset(
    "/work/bm1183/m301049/icon_hcap_data/plus4K/production/jed0022_atm_2d_hp.nc"
).pipe(egh.attach_coords)
iwp = ds["clivi"] + ds["qsvi"] + ds["qgvi"]
iwp_trop = iwp.where((ds.lat < 20) & (ds.lat > -20), drop=True)
# %%
iwp_stacked = iwp_trop.stack(idx=("time", "cell")).reset_index("idx")
# %% get random samples
sizes = [int(1e5), int(1e6), int(1e7)]
number = 7
random_samples = {}
for size in sizes:
    samples = []
    for j in range(number):
        values = np.random.choice(iwp_stacked.idx, size=size)
        samples.append(iwp_stacked.sel(idx=values))
    random_samples[size] = samples


# %% get histograms
iwp_bins = np.logspace(-6, 1, 100)
histograms = {}
std = {}
mean = {}
for size in sizes:
    hists = []
    for i in range(number):
        hist, edges = np.histogram(
            random_samples[size][i], bins=iwp_bins, density=False
        )
        hist = hist / size
        hists.append(hist)
    histograms[size] = hists
    std[size] = np.std(hists, axis=0)
    mean[size] = np.mean(hists, axis=0)

# %% plot iwp distributions
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, sharey=True)

for i, size in enumerate(sizes):
    ax = axes[i]
    for j in range(number):
        ax.stairs(histograms[size][j], edges, color="grey", alpha=0.5)
    ax.set_title(f"Size: {size:.0e}")
    ax.stairs(mean[size], edges, color="black")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("$P(I)$")
ax.set_xscale("log")
ax.set_xlabel("$I$ / kg m$^{-2}$")
fig.savefig("plots/iwp_dist_sampling_error.png", dpi=300)

# %%
timeslices = {
    "5day": [
        slice("1979-09-01", "1979-09-05"),
        slice("1979-09-06", "1979-09-10"),
        slice("1979-09-11", "1979-09-15"),
        slice("1979-09-16", "1979-09-20"),
        slice("1979-09-21", "1979-09-25"),
        slice("1979-09-26", "1979-09-29"),
    ],
    "10day": [
        slice("1979-08-31", "1979-09-09"),
        slice("1979-09-10", "1979-09-19"),
        slice("1979-09-20", "1979-09-29"),
    ],
    "15day": [
        slice("1979-08-31", "1979-09-15"),
        slice("1979-09-16", "1979-09-29"),
    ],
}


size = int(1e7)
samples = {}
for timeslice in timeslices.keys():
    samples[timeslice] = []
    for i in range(len(timeslices[timeslice])):
        iwp_stacked_slice = (
            iwp_trop.sel(time=timeslices[timeslice][i])
            .stack(idx=("time", "cell"))
            .reset_index("idx")
        )
        values = np.random.choice(iwp_stacked_slice.idx, size=size)
        samples[timeslice].append(iwp_stacked_slice.sel(idx=values))

# %%
histograms = {}
means = {}
stds = {}
for timeslice in timeslices.keys():
    histograms[timeslice] = []
    for i in range(len(timeslices[timeslice])):
        hist, edges = np.histogram(
            samples[timeslice][i], bins=iwp_bins, density=False
        )
        hist = hist / size
        histograms[timeslice].append(hist)
    means[timeslice] = np.mean(histograms[timeslice], axis=0)
    stds[timeslice] = np.std(histograms[timeslice], axis=0)

# %%
fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True, sharey='col')

for i, timeslice in enumerate(timeslices.keys()):
    for j in range(len(timeslices[timeslice])):
        axes[i, 0].stairs(histograms[timeslice][j], edges, color='grey')
    axes[i, 0].set_title(f"Timeslice: {timeslice}")
    axes[i, 0].set_ylabel("$P(I)$")
    axes[i, 1].stairs(stds[timeslice], edges, color='black')
    axes[i, 1].stairs(-stds[timeslice], edges, color='black')
    axes[i, 1].set_title(f"Timeslice: {timeslice}")
    axes[i, 1].set_ylabel("$\sigma$ P(I)")

for ax in axes.flatten():
    ax.set_xscale("log")
    ax.spines[["top", "right"]].set_visible(False)

for ax in axes[-1, :]:
    ax.set_xlabel("$I$ / kg m$^{-2}$")

fig.savefig("plots/iwp_dist_variability_4K.png", dpi=300, bbox_inches='tight')

# %%
