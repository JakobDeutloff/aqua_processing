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
)
from scipy.signal import savgol_filter

# %%
runs = ["jed0011", "jed0022", "jed0033"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
colors = {"jed0011": "k", "jed0022": "r", "jed0033": "orange"}
datasets = {}
datasets
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_tgrid_20.nc"
    ).sel(temp=slice(200, None))
    ds = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_20.nc"
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
    ) < 1e-1

# %% calculte stability iris parameters from instantaneous values
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


# calcualte sub and conv from mean values
hrs_mean = {}
sub_mean = {}
conv_mean = {}
sub_mean_cont = {}
conv_mean_cont = {}
mean_hrs_control = hrs["jed0011"].where(masks_clearsky["jed0011"]).median("index")
for run in runs:
    hrs_mean[run] = hrs[run].where(masks_clearsky[run]).median("index")
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
# %% plot results jevanjee
fig, axes = plt.subplots(1, 4, figsize=(14, 6), sharey=True)

for run in runs:
    axes[0].plot(
        hrs[run]["net_hr"].where(masks_clearsky[run]).median("index"),
        hrs[run]["temp"],
        label=exp_name[run],
        color=colors[run],
    )
    axes[1].plot(
        hrs[run]["stab"].where(masks_clearsky[run]).median("index"),
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
fig.savefig("plots/iwp_drivers/stab_iris_temp_20.png", dpi=300, bbox_inches="tight")

# %%
datasets["jed0011"]["ta"].astype(float).std('index').plot.line(x="temp", color="k", add_legend=False)

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
        f_conv[run].where(masks_clearsky[run]).mean("index"),
        f_conv[run]["temp"],
        label=exp_name[run],
        color=colors[run],
    )
    axes[1].plot(
        mean_rho[run],
        mean_rho[run]["temp"],
        label=exp_name[run],
        color=colors[run],
    )
    axes[2].plot(
        mean_hr[run],
        mean_hr[run]["temp"],
        label=exp_name[run],
        color=colors[run],
    )


axes[0].invert_yaxis()
axes[0].set_ylabel("Temperature / K")
axes[0].set_xlabel("Net flux convergence / W m$^{-3}$")
axes[1].set_xlabel("Air Density / kg m$^{-3}$")
axes[2].set_xlabel("Heating rate / K day$^{-1}$")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
)
fig.savefig("plots/iwp_drivers/flux_conv_temp.png", dpi=300, bbox_inches="tight")


# %%
def calc_rel_hum(hus, ta, p):
    e_sat = 6.112 * np.exp((22.46 * (ta - 273.15)) / ((ta - 273.15) + 272.62))
    e = hus * p / (0.622 + hus)
    return e / e_sat


def convert_to_densiy(var, dry_air, rho_air):
    var_dens = (var / dry_air) * rho_air
    return var_dens


rel_hum = {}
spec_hum = {}
dens = {}
for run in runs:
    rel_hum[run] = calc_rel_hum(
        datasets[run]["hus"],
        datasets[run]["ta"],
        datasets[run]["pfull"],
    )
    dens[run] = convert_to_densiy(
        datasets[run]["hus"],
        datasets[run]["pfull"] / (287.04 * datasets[run]["ta"]),
        1
        - (
            datasets[run]["cli"]
            + datasets[run]["clw"]
            + datasets[run]["qs"]
            + datasets[run]["qg"]
            + datasets[run]["qr"]
            + datasets[run]["hus"]
        ),
    )
    rel_hum[run] = rel_hum[run].where(masks_clearsky[run]).mean("index")
    spec_hum[run] = datasets[run]["hus"].where(masks_clearsky[run]).mean("index")

# %% plot absolute humidity
fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharey=True)


for run in runs:
    axes[0, 0].plot(
        rel_hum[run],
        rel_hum[run]["temp"],
        label=exp_name[run],
        color=colors[run],
    )

    axes[0, 1].plot(
        spec_hum[run],
        spec_hum[run]["temp"],
        label=exp_name[run],
        color=colors[run],
    )

    if run in ["jed0022", "jed0033"]:

        axes[1, 0].plot(
            (rel_hum[run] - rel_hum["jed0011"]) / rel_hum["jed0011"] * 100,
            rel_hum["jed0011"]["temp"],
            label=exp_name[run],
            color=colors[run],
            linestyle="-",
        )

        axes[1, 1].plot(
            (spec_hum[run] - spec_hum["jed0011"]) / spec_hum["jed0011"] * 100,
            spec_hum["jed0011"]["temp"],
            label=exp_name[run],
            color=colors[run],
            linestyle="-",
        )


for ax in axes.flatten():
    ax.spines[["top", "right"]].set_visible(False)

axes[0, 0].set_xlabel("Relative humidity / %")
axes[0, 1].set_xlabel("Specific humidity / kg kg$^{-1}$")
axes[1, 0].set_xlabel("Relative humidity difference / %")
axes[1, 1].set_xlabel("Specific humidity difference / %")

for ax in axes[:, 0]:
    ax.set_ylabel("Temperature / K")
    ax.invert_yaxis()
    ax.set_ylim([260, 213])

axes[0, 1].set_xscale("log")
fig.tight_layout()
fig.savefig("plots/iwp_drivers/humidity_temp.png", dpi=300)

# %% plot ozone
fig, ax = plt.subplots(figsize=(4, 6))
for run in runs:
    ax.plot(
        datasets[run]["o3"].mean("index"),
        datasets[run]["temp"],
        label=exp_name[run],
        color=colors[run],
    )

ax.set_xlabel("Ozone / kg kg$^{-1}$")
ax.set_ylabel("Temperature / K")
ax.spines[["top", "right"]].set_visible(False)
ax.invert_yaxis()
fig.savefig("plots/iwp_drivers/ozone_temp.png", dpi=300, bbox_inches="tight")

# %%
