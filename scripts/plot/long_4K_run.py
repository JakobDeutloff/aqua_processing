# %% 
import numpy as np 
import xarray as xr 
import matplotlib.pyplot as plt
from src.grid_helpers import merge_grid, fix_time

# %% load data 
ds_first_month = xr.open_mfdataset(
    "/work/bm1183/m301049/icon-mpim-4K/experiments/jed0022/jed0022_atm_2d_19*.nc"
).pipe(merge_grid).pipe(fix_time)
ds_first_month =  ds_first_month.sel(
    time=(ds_first_month.time.dt.minute == 0) & (ds_first_month.time.dt.hour == 0)
)
ds_last_two_months = xr.open_mfdataset(
    "/work/bm1183/m301049/icon-mpim-4K/experiments/jed0222/jed0222_atm_2d_19*.nc"
).pipe(merge_grid).pipe(fix_time)
ds_last_two_months = ds_last_two_months.sel(
    time=(ds_last_two_months.time.dt.minute == 0) & (ds_last_two_months.time.dt.hour == 0)
)
ds = xr.concat([ds_first_month, ds_last_two_months], dim="time")

# %% calculate IWP 
iwp_trop = (ds['clivi'] + ds['qsvi'] + ds['qgvi']).where(
    (ds.clat < 20) & (ds.clat > -20), drop=True).load()

# %% check variability 
timeslices = {
"15_days": [
    slice('1979-08-31','1979-09-14'),
    slice('1979-09-15','1979-09-30'),
    slice('1979-10-01','1979-10-15'),
    slice('1979-10-16','1979-10-31'),
    slice('1979-11-01','1979-11-14'),
    slice('1979-11-15','1979-11-30')
    ],
"30_days": [
    slice('1979-08-31','1979-09-30'),
    slice('1979-10-01','1979-10-31'),
    slice('1979-11-01','1979-11-30'),
],
"45_days":[
    slice('1979-08-31','1979-10-14'),
    slice('1979-10-15','1979-11-30'),
]
}


size = int(1e7)
samples = {}
for timeslice in timeslices.keys():
    samples[timeslice] = []
    for i in range(len(timeslices[timeslice])):
        print(i)
        iwp_stacked_slice = (
            iwp_trop.sel(time=timeslices[timeslice][i])
            .stack(idx=("time", "ncells"))
            .reset_index("idx")
            .dropna("idx")
        )
        values = np.random.choice(iwp_stacked_slice.idx, size=size)
        samples[timeslice].append(iwp_stacked_slice.sel(idx=values))

# %%
histograms = {}
means = {}
stds = {}
iwp_bins = np.logspace(-4, np.log10(40), 51)
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
fig, axes = plt.subplots(3, 2, figsize=(15, 8), sharex=True, sharey='col')

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
    ax.set_xlim([1e-4, 40])

fig.savefig("plots/iwp_dist_variability_long_4K.png", dpi=300, bbox_inches='tight')

# %% check radiation 
sw_cre = (ds['rsut'] - ds['rsutws']).isel(time=1).where(
    (ds.clat < 20) & (ds.clat > -20), drop=True
)
sw_cre_binned = sw_cre.groupby_bins(
    iwp_trop.isel(time=1), iwp_bins).mean()

fig, ax = plt.subplots()

ax.plot(iwp_bins[:-1], sw_cre_binned, color='black')
ax.set_xscale("log")
ax.set_xlabel("$I$ / kg m$^{-2}$")

# %% check hemispheric symmetry
iwp_dist_nh = {}
iwp_dist_sh = {}
size = int(1e6)
timeframes = {
    "one_month": slice("1979-08-31", "1979-09-30"),
    "two_months": slice("1979-08-31", "1979-10-31"),
    "three_months": slice("1979-08-31", "1979-11-30"),
}
for timeframe in timeframes.keys():
    print(f"Calculating IWP distribution for {timeframe}")
    # SH
    iwp_sh = (
        iwp_trop.sel(time=timeframes[timeframe])
        .where(iwp_trop.clat < 0, drop=True)
        .stack(idx=("time", "ncells"))
        .reset_index("idx")
        .dropna("idx")
    )
    values = np.random.choice(iwp_sh.idx, size=size)
    iwp_dist_sh[timeframe], edges = np.histogram(
        iwp_sh.sel(idx=values), bins=iwp_bins, density=False
    )
    iwp_dist_sh[timeframe] = iwp_dist_sh[timeframe] / size
    # NH
    iwp_nh = (
        iwp_trop.sel(time=timeframes[timeframe])
        .where(iwp_trop.clat > 0, drop=True)
        .stack(idx=("time", "ncells"))
        .reset_index("idx")
        .dropna("idx")
    )
    values = np.random.choice(iwp_nh.idx, size=size)
    iwp_dist_nh[timeframe], edges = np.histogram(
        iwp_nh.sel(idx=values), bins=iwp_bins, density=False
    )
    iwp_dist_nh[timeframe] = iwp_dist_nh[timeframe] / size


# %% plot hemispheric symmetry
fig, axes = plt.subplots(2, 3, figsize=(15, 5), sharex=True, sharey="row")

for i, timeframe in enumerate(timeframes.keys()):
    axes[0, i].stairs(
        iwp_dist_nh[timeframe], edges, color="blue", label=f"NH"
    )
    axes[0, i].stairs(
        iwp_dist_sh[timeframe], edges, color="red", label=f"SH"
    )
    axes[0, i].set_title(f"Timeslice: {timeframe}")
    axes[1, i].stairs(
        iwp_dist_nh[timeframe] - iwp_dist_sh[timeframe], edges, color="black"
    )

for ax in axes.flatten():
    ax.set_xscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim([1e-4, 40])

for ax in axes[-1, :]:
    ax.set_xlabel("$I$ / kg m$^{-2}$")
    ax.axhline(0, color="grey", linestyle="-")

axes[0, 0].set_ylabel("$P(I)$")
axes[1, 0].set_ylabel("$P(I)$ NH - $P(I)$ SH")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels, ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.1),
)
fig.savefig("plots/variability/iwp_dist_hemispheric_symmetry_long_4K.png", dpi=300, bbox_inches='tight')

# %%
