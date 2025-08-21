# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from src.calc_variables import (
    calc_heating_rates_t,
    calc_stability_t,
    calc_w_sub_t,
    calc_conv_t,
    calc_pot_temp,
    calc_flux_conv_t,
    calc_heating_rates,
    calc_stability,
    calc_w_sub,
    calc_conv,
)
from scipy.signal import savgol_filter
from scipy.stats import linregress
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

# %% calculte stability iris parameters in /K
hrs = {}
conv = {}
for run in runs:
    print(run)
    datasets[run] = datasets[run].assign(
        theta=calc_pot_temp(datasets[run]["ta"], datasets[run]["pfull"])
    )
    hrs[run] = calc_heating_rates_t(
        datasets[run]["rho"],
        datasets[run]["rsd"] - datasets[run]["rsu"],
        datasets[run]["rld"] - datasets[run]["rlu"],
        datasets[run]["zg"],
    )
    hrs[run] = hrs[run].assign(
        stab=calc_stability_t(
            datasets[run]["theta"],
            datasets[run]["ta"],
            datasets[run]["zg"],
        )
    )


#  calcualte sub and conv from mean values
hrs_mean = {}
sub_mean = {}
conv_mean = {}
sub_mean_cont = {}
conv_mean_cont = {}
mean_hrs_control = hrs["jed0011"].where(masks_clearsky["jed0011"]).mean("index")
for run in runs:
    hrs_mean[run] = hrs[run].where(masks_clearsky[run]).mean("index")
    sub_mean[run] = calc_w_sub_t(hrs_mean[run]["net_hr"], hrs_mean[run]["stab"])
    sub_mean[run] = xr.DataArray(
        data=savgol_filter(sub_mean[run], window_length=11, polyorder=3),
        coords=sub_mean[run].coords,
        dims=sub_mean[run].dims,
    )
    conv_mean[run] = calc_conv_t(sub_mean[run])
    conv_mean[run] = xr.DataArray(
        data=savgol_filter(conv_mean[run], window_length=11, polyorder=3),
        coords=conv_mean[run].coords,
        dims=conv_mean[run].dims,
    )

    if run in ["jed0022", "jed0033"]:
        sub_mean_cont[run] = calc_w_sub_t(
            mean_hrs_control["net_hr"], hrs_mean[run]["stab"]
        )
        sub_mean_cont[run] = xr.DataArray(
            data=savgol_filter(sub_mean_cont[run], window_length=11, polyorder=3),
            coords=sub_mean[run].coords,
            dims=sub_mean[run].dims,
        )
        conv_mean_cont[run] = calc_conv_t(sub_mean_cont[run])
        conv_mean_cont[run] = xr.DataArray(
            data=savgol_filter(conv_mean_cont[run], window_length=11, polyorder=3),
            coords=conv_mean_cont[run].coords,
            dims=conv_mean_cont[run].dims,
        )
# %% plot results /K
fig, axes = plt.subplots(1, 4, figsize=(14, 6), sharey=True)

for run in runs:
    axes[0].plot(
        hrs[run]["net_hr"].where(masks_clearsky[run]).mean("index"),
        hrs[run]["temp"],
        label=exp_name[run],
        color=colors[run],
    )
    axes[1].plot(
        hrs[run]["stab"].where(masks_clearsky[run]).mean("index"),
        hrs[run]["temp"],
        label=exp_name[run],
        color=colors[run],
    )
    axes[2].plot(
        sub_mean[run],
        sub_mean[run]["temp"],
        label=exp_name[run],
        color=colors[run],
    )
    axes[3].plot(
        -conv_mean[run],
        conv_mean[run]["temp"],
        label=exp_name[run],
        color=colors[run],
    )

for run in ["jed0022", "jed0033"]:
    axes[2].plot(
        sub_mean_cont[run],
        sub_mean_cont[run]["temp"],
        label=exp_name[run],
        color=colors[run],
        linestyle="--",
    )
    axes[3].plot(
        -conv_mean_cont[run],
        conv_mean_cont[run]["temp"],
        label=exp_name[run],
        color=colors[run],
        linestyle="--",
    )


for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)


axes[0].set_ylabel("Temperature / K")
axes[0].set_xlabel("Heating rate / K day$^{-1}$")
axes[1].set_xlabel("Stability / K K$^{-1}$")
axes[2].set_xlabel("Subsidence / K day$^{-1}$")
axes[3].set_xlabel("Convergence / day$^{-1}$")
axes[0].invert_yaxis()
axes[0].set_ylim([260, 200])
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
)
fig.savefig("plots/iwp_drivers/stab_iris_temp_K.png", dpi=300, bbox_inches="tight")

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
fig, axes = plt.subplots(1, 4, figsize=(14, 6), sharey=True)
plot_const_hr = True
for run in runs:
    axes[0].plot(
        hrs[run]["net_hr"].where(masks_clearsky[run]).mean("index"),
        hrs[run]["temp"],
        label=line_labels[run],
        color=colors[run],
    )
    axes[0].set_xlabel("Heating rate / K day$^{-1}$")
    axes[1].plot(
        stab[run].where(masks_clearsky[run]).mean("index"),
        stab[run]["temp"],
        label=line_labels[run],
        color=colors[run],
    )
    axes[1].set_xlabel("Stability / K m$^{-1}$")
    axes[2].plot(
        subs[run],
        subs[run]["temp"],
        label=line_labels[run],
        color=colors[run],
    )
    axes[2].set_xlabel("Subsidence / m day$^{-1}$")
    axes[3].plot(
        conv[run],
        conv[run]["temp"],
        label=line_labels[run],
        color=colors[run],
    )
    axes[3].set_xlabel("Convergence /  day$^{-1}$")
if plot_const_hr:
    for run in ["jed0022"]:
        axes[2].plot(
            subs_cont[run],
            subs_cont[run]["temp"],
            label=line_labels[run],
            color=colors[run],
            linestyle="--",
        )
        axes[3].plot(
            conv_cont[run],
            conv_cont[run]["temp"],
            label=line_labels[run],
            color=colors[run],
            linestyle="--",
        )

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
    ncol=3,
)
fig.tight_layout()
fig.savefig("plots/iwp_drivers/stab_iris_temp.png", dpi=300, bbox_inches="tight")

# %% make scatterplot of max convergence and Ts
max_conv = {}
t_delta = {"jed0011": 0, "jed0033": 2, "jed0022": 4,}
for run in ['jed0011', 'jed0033', 'jed0022']:
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
ax.set_xlabel("$\Delta T_s$ / K")
ax.set_ylabel(" $D_{\mathrm{max}}$ / day$^{-1}$")
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
fig.savefig(
    "plots/iwp_drivers/max_conv_comparison.png",
    dpi=300,
    bbox_inches="tight",
)


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
fig, axes = plt.subplots(1, 3, figsize=(10, 6), sharey=True)

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

handles, names = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    names,
    loc="center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
)
fig.tight_layout()
fig.savefig("plots/iwp_drivers/flux_conv_temp.png", dpi=300, bbox_inches="tight")


# %% look at stability changes between day and night 
stab_day = {}
stab_night = {}
for run in runs:
    mask_day = (datasets[run]["time_local"] > 6) & (datasets[run]['time_local']<18)
    stab_day[run] = stab[run].where(mask_day).mean("index")
    stab_night[run] = stab[run].where(~mask_day).mean("index")

fig, axes = plt.subplots(1, 3, figsize=(10, 6), sharey=True)

for run in runs:
    axes[0].plot(
        stab_day[run],
        stab_day[run]["temp"],
        label=line_labels[run],
        color=colors[run],
    )
    axes[1].plot(
        stab_night[run],
        stab_night[run]["temp"],
        label=line_labels[run],
        color=colors[run],
    )
    axes[2].plot(
        stab_day[run] - stab_night[run],
        stab_day[run]["temp"],
        label=line_labels[run],
        color=colors[run],
    )

# %%
