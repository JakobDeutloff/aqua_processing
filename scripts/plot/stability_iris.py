# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from src.calc_variables import (
    calc_heating_rates,
    calc_stability,
    calc_w_sub,
    calc_conv,
    calc_pot_temp,
)
from scipy.signal import savgol_filter

# %% load data
runs = ["jed0011", "jed0022", "jed0033"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
datasets = {}
datasets_tgrid = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample.nc"
    ).sel(index=slice(0, 1e5))
    datasets_tgrid[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_tgrid.nc"
    ).sel(temp=slice(213, None))
vgrid = (
    xr.open_dataset(
        "/work/bm1183/m301049/icon-mpim/experiments/jed0001/atm_vgrid_angel.nc"
    )
    .mean("ncells")
    .rename({"height": "height_2", "height_2": "height"})
)

# %% determine tropopause height and clearsky
idx_trop = {}
height_trop = {}
masks_clearsky = {}
mask_trop = {}
mask_stratosphere = vgrid["zg"].values < 20e3
for run in runs:
    idx_trop[run] = datasets[run]["ta"].where(mask_stratosphere).argmin("height")
    height_trop[run] = datasets[run]["height"].isel(height=idx_trop[run])
    masks_clearsky[run] = (
        datasets[run]["clivi"] + datasets[run]["qsvi"] + datasets[run]["qgvi"]
    ) < 1e-1
    mask_trop[run] = datasets[run]["height"] > height_trop[run]

# %% calculte stability iris parameters from instantaneous values
hrs = {}
conv = {}
for run in runs:
    print(run)
    datasets_tgrid[run] = datasets_tgrid[run].assign(
        theta=calc_pot_temp(datasets_tgrid[run]["ta"], datasets_tgrid[run]["pfull"])
    )
    hrs[run] = calc_heating_rates(
        datasets_tgrid[run]["rho"],
        datasets_tgrid[run]["rsd"] - datasets_tgrid[run]["rsu"],
        datasets_tgrid[run]["rld"] - datasets_tgrid[run]["rlu"],
        datasets_tgrid[run]["zg"],
    )
    hrs[run] = hrs[run].assign(
        stab=calc_stability(
            datasets_tgrid[run]["theta"],
            datasets_tgrid[run]["ta"],
            datasets_tgrid[run]["zg"],
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
    sub_mean[run] = calc_w_sub(hrs_mean[run]["net_hr"], hrs_mean[run]["stab"])
    sub_mean[run] = xr.DataArray(
        data=savgol_filter(sub_mean[run], window_length=11, polyorder=3),
        coords=sub_mean[run].coords,
        dims=sub_mean[run].dims,
    )
    conv_mean[run] = calc_conv(sub_mean[run])
    conv_mean[run] = xr.DataArray(
        data=savgol_filter(conv_mean[run], window_length=11, polyorder=3),
        coords=conv_mean[run].coords,
        dims=conv_mean[run].dims,
    )

    if run in ["jed0022", "jed0033"]:
        sub_mean_cont[run] = calc_w_sub(
            mean_hrs_control["net_hr"], hrs_mean[run]["stab"]
        )
        sub_mean_cont[run] = xr.DataArray(
            data=savgol_filter(sub_mean_cont[run], window_length=11, polyorder=3),
            coords=sub_mean[run].coords,
            dims=sub_mean[run].dims,
        )
        conv_mean_cont[run] = calc_conv(sub_mean_cont[run])
        conv_mean_cont[run] = xr.DataArray(
            data=savgol_filter(conv_mean_cont[run], window_length=11, polyorder=3),
            coords=conv_mean_cont[run].coords,
            dims=conv_mean_cont[run].dims,
        )

# %% plot results jevanjee
colors = {"jed0011": "k", "jed0022": "r", "jed0033": "orange"}
fig, axes = plt.subplots(1, 4, figsize=(14, 6), sharey=True)

for run in runs:
    mean_temp = (
        datasets[run]["ta"].where(mask_trop[run] & masks_clearsky[run]).mean("index")
    )
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
    mean_temp = (
        datasets[run]["ta"].where(masks_clearsky[run] & mask_trop[run]).mean("index")
    )
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
axes[0].set_ylim([260, 213])
fig.tight_layout()
fig.savefig("plots/iwp_drivers/stab_iris.png", dpi=300)

# %% compare actual lapse rates
laps_rate = {}
for run in runs:
    laps_rate[run] = np.abs(
        (
            (datasets[run]["ta"].diff("height") / vgrid["zg"].diff("height_2").values)
            .where(masks_clearsky[run] * mask_trop[run])
            .mean("index")
        )
    )

fig, ax = plt.subplots(figsize=(4, 6))
height_range = slice(6e3, 16e3)

for run in runs:
    mean_temp = (
        datasets[run]["ta"].where(masks_clearsky[run] & mask_trop[run]).mean("index")
    )
    ax.plot(
        laps_rate[run].where(mask_hrs),
        mean_temp.sel(height=laps_rate[run]["height"]).where(mask_hrs),
        label=exp_name[run],
        color=colors[run],
    )


ax.invert_yaxis()
ax.set_ylim([260, 200])
ax.set_xlim([0.004, 0.0085])
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylabel("Temperature / K")
ax.set_xlabel("Lapse rate / K m$^{-1}$")

# %% look at real clear-sky convergence
conv_real = {}
for run in ["jed0011", "jed0022"]:
    conv_real[run] = (
        (
            datasets[run]["wa"].diff("height_2")
            / vgrid["zghalf"][:-1].diff("height").values
        )
        .rename({"height_2": "height"})
        .where(masks_clearsky[run] & mask_trop[run])
        .mean("index")
    )

fig, ax = plt.subplots(1, 1, figsize=(4, 6))

for run in ["jed0011", "jed0022"]:
    mean_temp = (
        datasets[run]["ta"].where(masks_clearsky[run] & mask_trop[run]).mean("index")
    )
    ax.plot(
        conv_real[run].where(mask_conv) * 86400,
        mean_temp.sel(height=conv_real[run]["height"]).where(mask_conv),
        label=exp_name[run],
        color=colors[run],
    )
    ax.invert_yaxis()
    ax.set_ylim([260, 200])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("Temperature / K")
    ax.set_xlabel("Convergence / day$^{-1}$")

fig.savefig("plots/iwp_drivers/conv_real.png", dpi=300)


# %%
def calc_rel_hum(hus, ta, p):
    e_sat = 6.112 * np.exp((22.46 * (ta - 273.15)) / ((ta - 273.15) + 272.62))
    e = hus * p / (0.622 + hus)
    return e / e_sat


rel_hum = {}
spec_hum = {}
for run in runs:
    rel_hum[run] = calc_rel_hum(
        datasets_tgrid[run]["hus"],
        datasets_tgrid[run]["ta"],
        datasets_tgrid[run]["pfull"],
    )
    rel_hum[run] = rel_hum[run].where(masks_clearsky[run]).mean("index")
    spec_hum[run] = datasets_tgrid[run]["hus"].where(masks_clearsky[run]).mean("index")

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
fig.savefig("plots/iwp_drivers/humidity.png", dpi=300)

# %%
