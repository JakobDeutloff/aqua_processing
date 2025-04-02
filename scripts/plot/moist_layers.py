# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# %% load data
runs = ["jed0011", "jed0033", "jed0022"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
colors = {"jed0011": "k", "jed0022": "r", "jed0033": "orange"}
datasets = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_20.nc"
    ).sel(index=slice(0, 1e6))

vgrid = (
    xr.open_dataset(
        "/work/bm1183/m301049/icon-mpim/experiments/jed0001/atm_vgrid_angel.nc"
    )
    .mean("ncells")
    .rename({"height_2": "height", "height": "height_2"})
)
# %% calculate variance of v, q and covariance 
v_sq = {}
q_sq = {}
cov = {}
for run in runs:
    mask = datasets[run]['clat'] < 0 
    mean_v = datasets[run]['va'].where(mask).mean('index')
    mean_q = datasets[run]['hus'].where(mask).mean('index')
    v_sq[run] = np.abs(datasets[run]['va'].where(mask) - mean_v)
    q_sq[run] = np.abs(datasets[run]['hus'].where(mask) - mean_q) 
    cov[run] =((datasets[run]['va'].where(mask) - mean_v)) * ((datasets[run]['hus'].where(mask) - mean_q))

# %% plot variances
fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
for run in runs:
    axes[0].plot(
        v_sq[run].mean("index"),
        vgrid["zg"]/1e3,
        label=exp_name[run],
        color=colors[run],
    )

    axes[1].plot(
        q_sq[run].mean("index"),
        vgrid["zg"]/1e3,
        label=exp_name[run],
        color=colors[run],
    )

    axes[2].plot(
        cov[run].mean("index"),
        vgrid["zg"]/1e3,
        label=exp_name[run],
        color=colors[run],
    )

for ax in axes:
    ax.set_ylim(0, 18)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Height / km")
axes[0].set_xlabel(r"$\overline{|v'|}$ / m s$^{-1}$")
axes[0].set_xlim([0, 6])
axes[1].set_xlabel(r"$\overline{|q'|}$ / kg kg$^{-1}$")
axes[1].set_xlim([0, 0.0025])
axes[2].set_xlabel(r"$\overline{v' \cdot q'}$ / m kg s$^{-1}$ kg$^{-1}$")
#axes[2].set_xlim([0, 0.007])

fig.legend(
    handles=axes[0].lines,
    labels=[exp_name[run] for run in runs],
    bbox_to_anchor=(0.63, 0),
    ncols=3,
)
fig.tight_layout()
fig.savefig('plots/misc/qv_variance.png', dpi=300, bbox_inches='tight')

# %%
