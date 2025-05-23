# %%
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import matplotlib as mpl
from src.calc_variables import calc_IWC_cumsum, calculate_hc_temperature_bright

# %% load data
runs = ["jed0011", "jed0022", "jed0033"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
colors = {"jed0011": "k", "jed0022": "r", "jed0033": "orange"}
labels = {
    "jed0011": "Control",
    "jed0022": "+4 K",
    "jed0033": "+2 K",
}
datasets = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed.nc"
    )
vgrid = (
    xr.open_dataset(
        "/work/bm1183/m301049/icon-mpim/experiments/jed0001/atm_vgrid_angel.nc"
    )
    .mean("ncells")
    .rename({"height": "height_2", "height_2": "height"})
)

# %% calculate hc temp with brightness temperature
hc_temps = {}
for run in runs:
    print(run)
    temp, p = calculate_hc_temperature_bright(
        datasets[run],
        vgrid["zg"],
    )
    datasets[run] = datasets[run].assign({"hc_top_temp_bright": temp, "hc_top_pressure_bright": p})


# %% bin temperature
iwp_bins = np.logspace(-4, 1, 51)
binned_temp = {}
for run in runs:
    binned_temp[run] = (
        datasets[run]["hc_top_temp_bright"]
        .groupby_bins(datasets[run]["iwp"], iwp_bins)
        .mean()
    )

# %% plot hc_temp binned by IWP
fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex="col", height_ratios=[3, 1])

for run in runs:
    binned_temp[run].plot(
        ax=axes[0],
        color=colors[run],
        label=labels[run],
    )

axes[1].axhline(0, color="k", lw=0.5, ls="-")
for run in runs[1:]:
    (binned_temp[run] - binned_temp[runs[0]]).plot(
        ax=axes[1],
        color=colors[run],
        label=labels[run],
    )

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("")
    ax.set_xscale("log")
    ax.set_xlim([1e-4, 10])

axes[0].set_ylabel(r"T$_{\mathrm{hc}}$ / K")
axes[1].set_ylabel(r"T$_{\mathrm{hc}}$ - T$_{\mathrm{hc}}$ Control / K")
axes[1].set_xlabel(r"$I$ / kg m$^{-2}$")

handles, names = axes[0].get_legend_handles_labels()

fig.legend(
    handles=handles,
    labels=names,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.1),
    ncols=3,
)
fig.savefig(
    'plots/publication/fat.png', dpi=300, bbox_inches="tight"
)

# %% calculate mean hc temperatures
means = {}
for run in runs:
    means[run] = datasets[run]["hc_top_temp_bright"].where(datasets[run]["iwp"] > 1e-4).mean().values
    print(f"mean hc temp {exp_name[run]} {means[run]:.2f} K")

# %% plot hist of hc temps
fig, ax = plt.subplots(figsize=(8, 5))
medians = {}
for run in runs:
    hist, edges = np.histogram(
        datasets[run]["hc_top_temperature"].where((datasets[run]["iwp"] > 1e-4) & (datasets[run]["hc_top_temperature"] < 238)),
        bins=np.linspace(180, 238, 30),
        density=True
    )
    medians[run] = datasets[run]["hc_top_temperature"].where((datasets[run]["iwp"] > 1e-4) & (datasets[run]["hc_top_temperature"] < 238)).median().values
    ax.stairs(
        hist,
        edges,
        color=colors[run],
        label=labels[run],
    )
    ax.axvline(
        medians[run],
        color=colors[run],
        linestyle="--",
        lw=0.5,
    )
ax.set_xlabel("T$_{\mathrm{hc}}$ / K")


# %% calculate high cloud temperature weighted by cf 
cfs = {}
cfs_binned = {}
temp_binned = {}
for run in runs:
    cfs[run] = (
        datasets[run].sel(index=slice(0, 1e6))["cli"] + datasets[run].sel(index=slice(0, 1e6))["qs"] + datasets[run].sel(index=slice(0, 1e6))["qg"]
    )
    cfs_binned[run] = cfs[run].groupby_bins(datasets[run].sel(index=slice(0, 1e6))["iwp"], iwp_bins).mean()
    temp_binned[run] = (
        datasets[run].sel(index=slice(0, 1e6))["ta"]
        .groupby_bins(datasets[run].sel(index=slice(0, 1e6))["iwp"], iwp_bins)
        .mean()
    )

# %% weighted temperature 
t_weighted = {}
for run in runs:
    t_weighted[run] = (
        cfs[run] * datasets[run]['ta']
    ).sum("height") / cfs[run].sum("height")

# %% 
t_weighted_control = t_weighted['jed0011'].where(datasets['jed0011']['iwp']>1e-4).mean().values
t_hc_control = datasets['jed0011']['hc_top_temperature'].where(datasets['jed0011']['iwp']>1e-4).mean().values
for run in ['jed0033', 'jed0022']:
    print(f"weighted temperature {exp_name[run]} {(t_weighted[run].where(datasets[run]['iwp']>1e-4).mean().values - t_weighted_control):.2f} K")
for run in ['jed0033', 'jed0022']:
    print(f"T hc {exp_name[run]} {(datasets[run]['hc_top_temperature'].where(datasets[run]['iwp']>1e-4).mean().values - t_hc_control):.2f} K")
# %% plot t_weighted
fig, ax = plt.subplots(figsize=(8, 5))
for run in runs:
    t_weighted[run].groupby_bins(datasets[run]['iwp'].sel(index=slice(0, 1e6)), iwp_bins).mean().plot(
        ax=ax,
        color=colors[run],
    )
ax.set_xscale('log')
# %% contour binned cfs 
fig, ax = plt.subplots(figsize=(8, 5))
ax.contourf(
    iwp_bins[:-1],
    vgrid["zg"].values/1e3,
    cfs['jed0011'].T

)
ax.set_xscale('log')
ax.set_ylim([0, 20])


# %%
ds_interp = {}
z_grid = np.linspace(vgrid['zg'].max().values, vgrid['zg'].min().values, 600)
height_grid = np.flip(np.interp(np.flip(z_grid), np.flip(vgrid['zg'].values), np.flip(datasets['jed0011']['height'].values)))
for run in runs:
    ds_interp[run] = datasets[run][['ta', 'cli', 'qs', 'qg', 'phalf']].sel(index=slice(0, 1e6)).interp(height=height_grid, method='linear')
    ds_interp[run]['dzghalf'] = np.abs(np.diff(z_grid).mean())
    ds_interp[run] = ds_interp[run].assign(iwc_cumsum=calc_IWC_cumsum(ds_interp[run]))
    # calc hc temp 
    ds_interp[run]['hc_top_temperature'], ds_interp[run]['hc_top_pressure'] = (
    calculate_hc_temperature(ds_interp[run], IWP_emission=0.06)
)


# %% plot hc temperature 
mpl.use('WebAgg')
plt.close('all')
fig, ax = plt.subplots(figsize=(8, 5))
for run in runs:
    ds_interp[run]['hc_top_temperature'].groupby_bins(datasets[run].sel(index=slice(0, 1e6))['iwp'], iwp_bins).mean().plot(
        ax=ax,
        color=colors[run],
    )
ax.set_xscale('log')
plt.show()

# %% plot diff between hc temperatures 
fig, ax = plt.subplots(figsize=(8, 5))
for run in runs:
    (ds_interp[run]['hc_top_temperature'].groupby_bins(datasets[run].sel(index=slice(0, 1e6))['iwp'], iwp_bins).mean() - datasets[run]['hc_top_temperature'].sel(index=slice(0, 1e6)).groupby_bins(datasets[run].sel(index=slice(0, 1e6))['iwp'], iwp_bins).mean()).plot(
        ax=ax,
        color=colors[run],
    )
ax.set_xscale('log')
plt.show()

# %%

fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharey=True)

i = 11
t_int = ds_interp['jed0011']['ta'].sel(index=i)
t = datasets['jed0011'].sel(index=slice(0, 1e6))['ta'].astype(float).sel(index=i)
iwp_cumsum_int = ds_interp['jed0011']['iwc_cumsum'].sel(index=i)
iwp_cumsum = datasets['jed0011']['iwc_cumsum'].sel(index=slice(0, 1e6)).astype(float).sel(index=i)

axes[0, 0].plot(t.sel(height=slice(40, None)), vgrid['zg'].sel(height=slice(40, None))/1e3, color='k')
axes[0, 0].plot(t_int.sel(height=slice(40, None)), z_grid[height_grid>=40]/1e3, color='r')

axes[0, 1].plot(iwp_cumsum.sel(height=slice(40, None)), vgrid['zg'].sel(height=slice(40, None))/1e3, color='k')
axes[0, 1].plot(iwp_cumsum_int.sel(height=slice(40, None)), z_grid[height_grid>=40]/1e3, color='r')

axes[1, 0].plot(datasets['jed0011']['cli'].sel(index=i).sel(height=slice(40, None)), vgrid['zg'].sel(height=slice(40, None))/1e3, color='k')
axes[1, 0].plot(ds_interp['jed0011']['cli'].sel(index=i).sel(height=slice(40, None)), z_grid[height_grid>=40]/1e3, color='r')

axes[1, 1].plot(datasets['jed0011']['qs'].sel(index=i).sel(height=slice(40, None)), vgrid['zg'].sel(height=slice(40, None))/1e3, color='k')
axes[1, 1].plot(ds_interp['jed0011']['qs'].sel(index=i).sel(height=slice(40, None)), z_grid[height_grid>=40]/1e3, color='r')

axes[1, 2].plot(datasets['jed0011']['qg'].sel(index=i).sel(height=slice(40, None)), vgrid['zg'].sel(height=slice(40, None))/1e3, color='k')
axes[1, 2].plot(ds_interp['jed0011']['qg'].sel(index=i).sel(height=slice(40, None)), z_grid[height_grid>=40]/1e3, color='r')

plt.show()



# %% plot grid spacing vs height 
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(vgrid['dzghalf'], vgrid['zg']/1e3)

plt.show()
# %%
