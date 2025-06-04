# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from src.calc_variables import (
    calc_heating_rates,
    calc_stability,
    calc_w_sub,
    calc_conv,
)
from scipy.signal import savgol_filter
from scipy.stats import linregress


# %%
runs = ["jed0011", "jed0033", "jed0022"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
colors = colors = {"jed0011": "#3e0237", "jed0022": "#f707da", "jed0033": "#9a0488"}
labels = {
    "jed0011": "Control",
    "jed0022": "+4 K",
    "jed0033": "+2 K",
}
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

# %% determine tropopause height and clearsky
masks_clearsky = {}
for run in runs:
    masks_clearsky[run] = (
        datasets[run]["clivi"] + datasets[run]["qsvi"] + datasets[run]["qgvi"]
    ) < 1e-2

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
fig, axes = plt.subplots(1, 4, figsize=(12, 5), sharey=True)
plot_const_hr = True
for run in runs:
    axes[0].plot(
        hrs[run]["net_hr"].where(masks_clearsky[run]).mean("index"),
        hrs[run]["temp"],
        label=labels[run],
        color=colors[run],
    )
    axes[0].set_xlabel("Heating Rate / K day$^{-1}$")
    axes[1].plot(
        stab[run].where(masks_clearsky[run]).mean("index"),
        stab[run]["temp"],
        label=labels[run],
        color=colors[run],
    )
    axes[1].set_xlabel("Stability / K m$^{-1}$")
    axes[2].plot(
        subs[run],
        subs[run]["temp"],
        label=labels[run],
        color=colors[run],
    )
    axes[2].set_xlabel("Subsidence / m day$^{-1}$")
    axes[3].plot(
        conv[run],
        conv[run]["temp"],
        label=labels[run],
        color=colors[run],
    )
    axes[3].set_xlabel("Convergence /  day$^{-1}$")
if plot_const_hr:
    for run in ["jed0022"]:
        axes[2].plot(
            subs_cont[run],
            subs_cont[run]["temp"],
            label=labels[run],
            color=colors[run],
            linestyle="--",
        )
        axes[3].plot(
            conv_cont[run],
            conv_cont[run]["temp"],
            color=colors[run],
            linestyle="--",
            label="+4 K Constant HR"
        )

axes[0].set_ylim([260, 200])
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Temperature / K")
axes[0].set_yticks([260, 230, 215, 200])
handles, names = axes[-1].get_legend_handles_labels()
fig.legend(
    handles,
    names,
    loc="center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=4,
    frameon=False,
)
# add letters
for i, ax in enumerate(axes):
    ax.text(
        0.03,
        1, 
        chr(97 + i),
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
    )

fig.tight_layout()
fig.savefig("plots/publication/figure4.pdf", bbox_inches="tight")

# %% make scatterplot of max convergence and Ts
max_conv = {}
t_delta = {"jed0011": 0, "jed0033": 2, "jed0022": 4,}
for run in runs:
    max_conv[run] = float(conv[run].max(dim="temp").values)

linreg = linregress(
    list(t_delta.values()),
    list(max_conv.values()),
)

fig, ax = plt.subplots(figsize=(4, 4))
for run in runs:
    ax.scatter(
        t_delta[run],
        max_conv[run],
        color=colors[run],
    )
ax.plot(
    list(t_delta.values()),
    [linreg.intercept + linreg.slope * t for t in t_delta.values()],
    color="grey",
)
ax.set_xticks(list(t_delta.values()))
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel(r"$\Delta T_s$ / K")
ax.set_ylabel(r"$D_{\mathrm{max}}$ / day$^{-1}$")
fig.tight_layout()
fig.savefig("plots/iwp_drivers/max_conv.png", dpi=300, bbox_inches="tight")

# %% make comparison plot to other studies
delta_dr = {
    "Jeevanjee (2022)": (0.65 - 0.27),
    "Bony et al. (2016)": (0.59 - 0.2),
    "Saint-Lu et al. (2020)": (0.006 + 0.01),
}
delta_t = {
    "Jeevanjee (2022)": (310 - 280),
    "Bony et al. (2016)": (310 - 285),
    "Saint-Lu et al. (2020)": (0.23 + 0.42),
}
slope = {}
for key in delta_dr.keys():
    slope[key] = -delta_dr[key] / delta_t[key]

fig, ax = plt.subplots(figsize=(2, 4))
for key in delta_dr.keys():

    ax.text(
        0.1,
        slope[key],
        key,
        fontsize=12,
    )
ax.text(
    0.1,
    linreg.slope + 0.001,
    "ICON",
    fontsize=12,
)
ax.set_ylim([0, -0.028])

ax.set_xticks([])
yticks = list(slope.values())
yticks.append(linreg.slope)
yticks.append(0)
yticks = np.round(yticks, 3)
ax.set_yticks(yticks)
ax.spines[["top", "right", "bottom"]].set_visible(False)
ax.set_ylabel(r"$\Delta D_{\mathrm{max}} /\Delta T_s$ / day$^{-1}$ K$^{-1}$") 

# %% plot real clear sky convergence 


