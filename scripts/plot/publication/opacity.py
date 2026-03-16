# %%
import matplotlib.pyplot as plt
import pickle
from src.read_data import load_definitions
import xarray as xr

# %%
results = {}
runs, exp_name, colors, line_labels, sw_color, lw_color, net_color, linestyles = (
    load_definitions()
)
for run in runs:
    results[run] = {}
    with open(f"data/{run}_hc_albedo.pkl", "rb") as f:
        results[run]["albedo"] = pickle.load(f)
    with open(f"data/{run}_hc_emissivity.pkl", "rb") as f:
        results[run]["emissivity"] = pickle.load(f)
# %% plot albedo and emissivity
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
for run in ["jed0022", "jed0033", "jed0011"]:
    axes[0].plot(
        results[run]["albedo"].index,
        results[run]["albedo"],
        color=colors[run],
        label=line_labels[run],
        linestyle=linestyles[run],
    )
    axes[1].plot(
        results[run]["albedo"].index,
        results[run]["emissivity"],
        color=colors[run],
        label=line_labels[run],
        linestyle=linestyles[run],
    )

for ax in axes:
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("$I$ / kg m$^{-2}$")

axes[1].set_ylabel("Emissivity")
axes[0].set_ylabel("Albedo")

# put letters
for i, ax in enumerate(axes):
    ax.text(
        0.1,
        0.97,
        chr(97 + i),
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.65, 0), ncol=3, ncols=3, frameon=False)
fig.savefig('plots/publication/albedo_emissivity.pdf', bbox_inches='tight')
# %% save data 
path ='/work/bm1183/m301049/icon_hcap_data/publication/opacity'
for run in runs:
    albedo = xr.DataArray(
        results[run]["albedo"].values,
        coords={"iwp_bins": results[run]["albedo"].index.values},
        dims=["iwp_bins"],
    )
    emissivity = results[run]["emissivity"]
    albedo.to_netcdf(f"{path}/{run}_albedo.nc")
    emissivity.to_netcdf(f"{path}/{run}_emissivity.nc")

# %%
