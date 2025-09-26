# %%
import xarray as xr
import matplotlib.pyplot as plt
from src.calc_variables import (
    calc_heating_rates,
    calc_stability,
    calc_w_sub,
    calc_conv,
)
from scipy.signal import savgol_filter
from src.read_data import load_definitions, load_random_datasets, load_vgrid

# %%
runs, exp_name, colors, line_labels, sw_color, lw_color, net_color, linestyles = load_definitions()
linestyles = {
    "jed0011": "-",
    "jed0022": "-",
    "jed0033": "-",
    "jed2224": "--",}
datasets = {}
datasets
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_tgrid_20.nc"
    ).sel(temp=slice(200, None))
    ds = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample.nc"
    ).sel(index=slice(0, 1e6))
    # Assign all variables from ds to datasets if dim == index
    datasets[run] = datasets[run].assign(
        **{var: ds[var] for var in ds.variables if ("index",) == ds[var].dims}
    )

datasets_rand = load_random_datasets()
datasets_rand["jed2224"] = xr.open_dataset(
    "/work/bm1183/m301049/icon_hcap_data/const_o3/production/random_sample/jed2224_randsample_processed_64.nc"
)
v_grid = load_vgrid()
# %% add const ozone run
runs = runs + ["jed2224"]
datasets["jed2224"] = xr.open_dataset(
    "/work/bm1183/m301049/icon_hcap_data/const_o3/production/random_sample/jed2224_randsample_tgrid_20.nc"
).sel(temp=slice(200, None))
ds = xr.open_dataset(
    "/work/bm1183/m301049/icon_hcap_data/const_o3/production/random_sample/jed2224_randsample.nc"
).sel(index=slice(0, 1e6))
datasets["jed2224"] = datasets["jed2224"].assign(
    **{var: ds[var] for var in ds.variables if ("index",) == ds[var].dims}
)
colors["jed2224"] = "k"
line_labels["jed2224"] = "+4 K const. O3"

# %% determine tropopause height and clearsky
masks_clearsky = {}
for run in runs:
    masks_clearsky[run] = (
        datasets[run]["clivi"] + datasets[run]["qsvi"] + datasets[run]["qgvi"]
    ) < 1e-1

# %% calculate stability iris parameters in /m
hrs = {}
stab = {}
subs = {}
subs_cont = {}
conv = {}
conv_cont = {}
for run in runs:
    print(run)
    hrs[run] = calc_heating_rates(
        datasets[run]["rho"],
        datasets[run]["rsd"] - datasets[run]["rsu"],
        datasets[run]["rld"] - datasets[run]["rlu"],
        datasets[run]["zg"],
    )
    stab[run] = calc_stability(datasets[run]["ta"], datasets[run]["zg"])

for run in runs:
    subs[run] = calc_w_sub(
        hrs[run]["net_hr"].where(masks_clearsky[run]).mean("index"),
        stab[run].where(masks_clearsky[run]).mean("index"),
    )
    subs[run] = xr.DataArray(
        data=savgol_filter(subs[run], window_length=11, polyorder=3),
        coords=subs[run].coords,
        dims=subs[run].dims,
    )
    conv[run] = calc_conv(
        subs[run],
        datasets[run]["zg"].where(masks_clearsky[run]).mean("index"),
    )
for run in ["jed0022", "jed0033"]:
    subs_cont[run] = calc_w_sub(
        hrs["jed0011"]["net_hr"].where(masks_clearsky["jed0011"]).mean("index"),
        stab[run].where(masks_clearsky[run]).mean("index"),
    )
    subs_cont[run] = xr.DataArray(
        data=savgol_filter(subs_cont[run], window_length=11, polyorder=3),
        coords=subs_cont[run].coords,
        dims=subs_cont[run].dims,
    )
    conv_cont[run] = calc_conv(
        subs_cont[run],
        datasets[run]["zg"].where(masks_clearsky[run]).mean("index"),
    )

# %% plot results in /m
fig, axes = plt.subplots(1, 4, figsize=(10, 4), sharey=True)
for run in runs:
    axes[0].plot(
        hrs[run]["net_hr"].where(masks_clearsky[run]).mean("index"),
        hrs[run]["temp"],
        label=line_labels[run],
        color=colors[run],
        linestyle=linestyles[run],
    )
    axes[0].set_xlabel("Heating rate / K day$^{-1}$")
    axes[1].plot(
        stab[run].where(masks_clearsky[run]).mean("index"),
        stab[run]["temp"],
        label=line_labels[run],
        color=colors[run],
        linestyle=linestyles[run],
    )
    axes[1].set_xlabel("Stability / K m$^{-1}$")
    axes[2].plot(
        subs[run],
        subs[run]["temp"],
        label=line_labels[run],
        color=colors[run],
        linestyle=linestyles[run],
    )
    axes[2].set_xlabel("Subsidence / m day$^{-1}$")
    axes[3].plot(
        conv[run],
        conv[run]["temp"],
        label=line_labels[run],
        color=colors[run],
        linestyle=linestyles[run],
    )
    axes[3].set_xlabel("Convergence /  day$^{-1}$")

axes[0].set_ylim([260, 200])
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Temperature / K")
handles, names = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    names,
    loc="center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=4,
)
fig.tight_layout()
fig.savefig("plots/publication/stab_iris_profiles_ozone.pdf", bbox_inches="tight")


# %% plot temperature profiles in clearsky 
masks_clearsky_rand = {}
for run in runs:
    masks_clearsky_rand[run] = (
        datasets_rand[run]["clivi"] + datasets_rand[run]["qsvi"] + datasets_rand[run]["qgvi"]
    ) < 1e-1

t_profile = {}
for run in runs:
    t_profile[run] = datasets_rand[run]["ta"].where(masks_clearsky_rand[run]).mean("index")
fig, ax = plt.subplots(figsize=(4, 6))
for run in runs:
    ax.plot(
        t_profile[run],
        v_grid["zg"]/1e3,
        label=line_labels[run],
        color=colors[run],
        linestyle=linestyles[run],
    )

ax.set_ylim([10, 20])
ax.set_xlim([190, 250])
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("Temperature / K")
ax.set_ylabel("Height / km")
ax.legend(
    ncol=1,
    fontsize=10,
)
fig.savefig("plots/publication/t_profile_clearsky.pdf", bbox_inches="tight")

# %%
