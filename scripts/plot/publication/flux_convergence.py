# %%
import xarray as xr
import matplotlib.pyplot as plt
from src.calc_variables import calc_flux_conv_t
from src.read_data import load_definitions

# %%
runs, exp_name, colors, line_labels, sw_color, lw_color, net_color, linestyles = load_definitions()
datasets = {}
datasets
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_tgrid_20.nc"
    ).sel(temp=slice(200, None))
    ds = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed_64.nc"
    ).sel(index=slice(0, 1e6))
    # Assign all variables from ds to datasets if dim == index
    datasets[run] = datasets[run].assign(
        **{var: ds[var] for var in ds.variables if ("index",) == ds[var].dims}
    )

# %% determine tropopause height and clearsky
masks_clearsky = {}
for run in runs:
    masks_clearsky[run] = (
        datasets[run]["clivi"] + datasets[run]["qsvi"] + datasets[run]["qgvi"]
    ) < 1e-2
# %% calculate convergence of net flux
f_conv = {}
mean_rho = {}
mean_hr = {}
for run in runs:
    f_conv[run] = (
        calc_flux_conv_t(
            (datasets[run]["rsd"] - datasets[run]["rsu"])
            + (datasets[run]["rld"] - datasets[run]["rlu"]),
            datasets[run]["zg"],
        )
        .where(masks_clearsky[run])
        .mean("index")
    )
    mean_rho[run] = datasets[run]["rho"].where(masks_clearsky[run]).mean("index")
    mean_hr[run] = (f_conv[run] * 86400) / (mean_rho[run] * 1004)
# %% plot convergence of net flux
fig, axes = plt.subplots(1, 3, figsize=(8, 4), sharey=True)

for run in runs:

    axes[0].plot(
        mean_hr[run],
        mean_hr[run]["temp"],
        label=line_labels[run],
        color=colors[run],
    )
    axes[2].plot(
        mean_rho[run],
        mean_rho[run]["temp"],
        label=line_labels[run],
        color=colors[run],
    )
    axes[1].plot(
        f_conv[run].where(masks_clearsky[run]).mean("index"),
        f_conv[run]["temp"],
        label=line_labels[run],
        color=colors[run],
    )



axes[0].invert_yaxis()
axes[0].set_ylabel("Temperature / K")
axes[1].set_xlabel("Net flux divergence / W m$^{-3}$")
axes[2].set_xlabel("Air Density / kg m$^{-3}$")
axes[0].set_xlabel("Heating rate / K day$^{-1}$")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
axes[1].set_xticks([-0.01, -0.005, 0])

handles, names = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    names,
    loc="center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
)
# add letters 
for ax, letter in zip(axes, ["a", "b", "c"]):
    ax.text(
        0.03,
        1,
        letter,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
    )
fig.tight_layout()
fig.savefig("plots/publication/flux_conv.pdf", bbox_inches="tight")
# %%
