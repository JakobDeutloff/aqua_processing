# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from src.calc_variables import (
    calc_heating_rates,
    calc_stability,
    calc_w_sub,
    calc_conv,
    calc_cf,
    calc_stability_jev,
    calc_w_sub_jev,
    calc_conv_jev,
)

# %% load data
runs = ["jed0011", "jed0022", "jed0033"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
datasets = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample.nc"
    ).sel(index=slice(0, 5e5))

vgrid = xr.open_dataset(
    "/work/bm1183/m301049/icon-mpim/experiments/jed0001/atm_vgrid_angel.nc"
).mean("ncells")


# %% calculate instataneous values
hrs = {}
masks_clearsky = {}
conv = {}
for run in runs:
    print(run)
    hrs[run] = calc_heating_rates(datasets[run], vgrid)
    hrs[run] = hrs[run].assign(stab=calc_stability(datasets[run]))
    hrs[run] = hrs[run].assign(sub=calc_w_sub(hrs[run]))
    conv[run] = calc_conv(
        hrs[run]["sub"], datasets[run]["pfull"].sel(height=hrs[run]["height"])
    )
    datasets[run] = datasets[run].assign(cf=calc_cf(datasets[run]))

    masks_clearsky[run] = (
        datasets[run]["clivi"] + datasets[run]["qsvi"] + datasets[run]["qgvi"]
    ) < 1e-1

# %% plot mean heating rate
fig, axes = plt.subplots(1, 5, figsize=(14, 6), sharey=True)
colors = {"jed0011": "k", "jed0022": "r", "jed0033": "orange"}
height_range = slice(6e3, 18e3)

for run in runs:
    mean_temp = datasets[run]["ta"].where(masks_clearsky[run]).mean("index")
    axes[0].plot(
        hrs[run]["net_hr"]
        .where(masks_clearsky[run])
        .mean("index")
        .where(
            (vgrid["zghalf"].sel(height=hrs[run]["height"]) >= height_range.start)
            & (vgrid["zghalf"].sel(height=hrs[run]["height"]) <= height_range.stop)
        ),
        mean_temp.sel(height=hrs[run]["height"]).where(
            (vgrid["zghalf"].sel(height=hrs[run]["height"]) >= height_range.start)
            & (vgrid["zghalf"].sel(height=hrs[run]["height"]) <= height_range.stop)
        ),
        label=exp_name[run],
        color=colors[run],
    )
    axes[1].plot(
        hrs[run]["stab"]
        .where(masks_clearsky[run])
        .mean("index")
        .where(
            (vgrid["zghalf"].sel(height=hrs[run]["height"]) >= height_range.start)
            & (vgrid["zghalf"].sel(height=hrs[run]["height"]) <= height_range.stop)
        ),
        mean_temp.sel(height=hrs[run]["height"]).where(
            (vgrid["zghalf"].sel(height=hrs[run]["height"]) >= height_range.start)
            & (vgrid["zghalf"].sel(height=hrs[run]["height"]) <= height_range.stop)
        ),
        label=exp_name[run],
        color=colors[run],
    )
    axes[2].plot(
        hrs[run]["sub"]
        .where(masks_clearsky[run])
        .median("index")
        .where(
            (vgrid["zghalf"].sel(height=hrs[run]["height"]) >= height_range.start)
            & (vgrid["zghalf"].sel(height=hrs[run]["height"]) <= height_range.stop)
        ),
        mean_temp.sel(height=hrs[run]["height"]).where(
            (vgrid["zghalf"].sel(height=hrs[run]["height"]) >= height_range.start)
            & (vgrid["zghalf"].sel(height=hrs[run]["height"]) <= height_range.stop)
        ),
        label=exp_name[run],
        color=colors[run],
    )
    axes[3].plot(
        conv[run]
        .where(masks_clearsky[run])
        .median("index")
        .where(
            (vgrid["zghalf"].sel(height=conv[run]["height"]) >= height_range.start)
            & (vgrid["zghalf"].sel(height=conv[run]["height"]) <= height_range.stop)
        ),
        mean_temp.sel(height=conv[run]["height"]).where(
            (vgrid["zghalf"].sel(height=conv[run]["height"]) >= height_range.start)
            & (vgrid["zghalf"].sel(height=conv[run]["height"]) <= height_range.stop)
        ),
        label=exp_name[run],
        color=colors[run],
    )
    axes[4].plot(
        datasets[run]["cf"]
        .mean("index")
        .where(
            (vgrid["zghalf"].sel(height=datasets[run]["height"]) >= height_range.start)
            & (vgrid["zghalf"].sel(height=datasets[run]["height"]) <= height_range.stop)
        ),
        mean_temp.sel(height=datasets[run]["height"]).where(
            (vgrid["zghalf"].sel(height=datasets[run]["height"]) >= height_range.start)
            & (vgrid["zghalf"].sel(height=datasets[run]["height"]) <= height_range.stop)
        ),
        label=exp_name[run],
        color=colors[run],
    )

axes[0].invert_yaxis()
axes[0].set_ylim([260, 200])

axes[0].set_ylabel("Temperature / K")
axes[0].set_xlabel("Heating rate / K day$^{-1}$")
axes[1].set_xlabel("Stability / mK hPa$^{-1}$")
axes[2].set_xlabel("Subsidence / hPa day$^{-1}$")
axes[3].set_xlabel("Convergence / day$^{-1}$")
axes[4].set_xlabel("Cloud Fraction")
axes[0].set_xlim([-1.5, 0.3])
axes[1].set_xlim([0, 200])
axes[2].set_xlim([-5, 35])
axes[3].set_xlim([-0.1, 0.4])

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

# %% determine tropopause height
height_trop = {}
mask_stratosphere = vgrid["zg"].values < 30e3
for run in runs:
    height_trop[run] = datasets[run]["ta"].where(mask_stratosphere).argmin("height")

# %% calculte jevanjee values
hrs_jev = {}
masks_clearsky = {}
mask_trop = {}
conv_jev = {}
for run in runs:
    mask_trop[run] = datasets[run]["height"] > height_trop[run]
    ds = datasets[run].where(mask_trop[run])
    hrs_jev[run] = calc_heating_rates(
        ds["rho"], ds["rsd"] - ds["rsu"], ds["rld"] - ds["rlu"], vgrid
    )
    hrs_jev[run] = hrs_jev[run].assign(stab=calc_stability_jev(ds["ta"], vgrid=vgrid))
    hrs_jev[run] = hrs_jev[run].assign(
        sub=calc_w_sub_jev(hrs_jev[run]["net_hr"], hrs_jev[run]["stab"])
    )
    conv_jev[run] = calc_conv_jev(hrs_jev[run]["sub"], vgrid)
    datasets[run] = datasets[run].assign(cf=calc_cf(datasets[run]))

    masks_clearsky[run] = (
        datasets[run]["clivi"] + datasets[run]["qsvi"] + datasets[run]["qgvi"]
    ) < 1e-1


# %% calcualte sub and conv from mean values
sub_mean = {}
sub_mean_cont = {}
conv_mean = {}
conv_mean_cont = {}
for run in runs:
    mean_hrs = hrs_jev[run].where(masks_clearsky[run]).mean("index")
    sub_mean[run] = calc_w_sub_jev(mean_hrs["net_hr"], mean_hrs["stab"])
    conv_mean[run] = calc_conv_jev(sub_mean[run], vgrid)

for run in ["jed0022", "jed0033"]:
    mean_hrs = hrs_jev[run].where(masks_clearsky[run]).mean("index")
    mean_hrs_control = hrs_jev["jed0011"].where(masks_clearsky["jed0011"]).mean("index")
    sub_mean_cont[run] = calc_w_sub_jev(mean_hrs_control["net_hr"], mean_hrs["stab"])
    conv_mean_cont[run] = calc_conv_jev(sub_mean_cont[run], vgrid)

# %% plot results jevanjee

colors = {"jed0011": "k", "jed0022": "r", "jed0033": "orange"}
fig, axes = plt.subplots(1, 4, figsize=(14, 6), sharey=True)
mask_hrs = (
    vgrid["zghalf"].sel(height=hrs_jev["jed0011"]["height"]) >= height_range.start
) & (vgrid["zghalf"].sel(height=hrs_jev["jed0011"]["height"]) <= height_range.stop)
mask_conv = (
    vgrid["zghalf"].sel(height=conv_jev["jed0011"]["height"]) >= height_range.start
) & (vgrid["zghalf"].sel(height=conv_jev["jed0011"]["height"]) <= height_range.stop)


for run in runs:
    mean_temp = (
        datasets[run]["ta"]
        .where(mask_trop[run] & masks_clearsky[run])
        .mean("index")
    )
    axes[0].plot(
        hrs_jev[run]["net_hr"]
        .where(masks_clearsky[run])
        .median("index")
        .where(mask_hrs),
        mean_temp.sel(height=hrs_jev[run]["height"]).where(mask_hrs),
        label=exp_name[run],
        color=colors[run],
    )
    axes[1].plot(
        hrs_jev[run]["stab"].where(masks_clearsky[run]).median("index").where(mask_hrs),
        mean_temp.sel(height=hrs_jev[run]["height"]).where(mask_hrs),
        label=exp_name[run],
        color=colors[run],
    )
    axes[2].plot(
        sub_mean[run].where(mask_hrs),
        mean_temp.sel(height=sub_mean[run]["height"]).where(mask_hrs),
        label=exp_name[run],
        color=colors[run],
    )
    axes[3].plot(
        -conv_mean[run].where(mask_conv),
        mean_temp.sel(height=conv_mean[run]["height"]).where(mask_conv),
        label=exp_name[run],
        color=colors[run],
    )

for run in ["jed0022", "jed0033"]:
    mean_temp = datasets[run]["ta"].where(masks_clearsky[run] & mask_trop[run]).mean("index")
    axes[2].plot(
        sub_mean_cont[run].where(mask_hrs),
        mean_temp.sel(height=sub_mean_cont[run]["height"]).where(mask_hrs),
        label=exp_name[run],
        color=colors[run],
        linestyle="--",
    )
    axes[3].plot(
        -conv_mean_cont[run].where(mask_conv),
        mean_temp.sel(height=conv_mean_cont[run]["height"]).where(mask_conv),
        label=exp_name[run],
        color=colors[run],
        linestyle="--",
    )


for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)


axes[0].set_ylabel("Temperature / K")
axes[0].set_xlabel("Heating rate / K day$^{-1}$")
axes[1].set_xlabel("Stability / K m$^{-1}$")
axes[2].set_xlabel("Subsidence / m day$^{-1}$")
axes[3].set_xlabel("Convergence / day$^{-1}$")
axes[0].set_xlim([-1.5, 0.3])
axes[1].set_xlim([-0.01, 0])
axes[0].invert_yaxis()
axes[0].set_ylim([260, 200])
fig.tight_layout()
fig.savefig("plots/iwp_drivers/stab_iris.png", dpi=300)


# %% determine tropopause height
fig, ax = plt.subplots(1, 1, figsize=(4, 6))
ax.plot(datasets["jed0011"]["ta"].mean("index"), vgrid["zg"], label="control")
ax.axhline(
    vgrid["zg"].sel(height_2=height_trop["jed0011"]).median(), color="k", linestyle="--"
)


# %% compare actual lapse rates to moist adiabats
laps_rate = {}
for run in runs:
    laps_rate[run] = np.abs(
        (
            (datasets[run]["ta"].diff("height") / vgrid["zg"].diff("height_2").values)
            .where(masks_clearsky[run])
            .mean("index")
        )
    )


# %% plot lapse rates
fig, ax = plt.subplots(figsize=(4, 6))
height_range = slice(6e3, 16e3)

for run in runs:
    mean_temp = datasets[run]["ta"].where(masks_clearsky[run]).mean("index")
    ax.plot(
        laps_rate[run].where(
            (vgrid["zghalf"].sel(height=laps_rate[run]["height"]) >= height_range.start)
            & (
                vgrid["zghalf"].sel(height=laps_rate[run]["height"])
                <= height_range.stop
            )
        ),
        mean_temp.sel(height=laps_rate[run]["height"]).where(
            (vgrid["zghalf"].sel(height=laps_rate[run]["height"]) >= height_range.start)
            & (
                vgrid["zghalf"].sel(height=laps_rate[run]["height"])
                <= height_range.stop
            )
        ),
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
        .where(masks_clearsky[run])
        .mean("index")
    )

# %% plot real convergence

fig, ax = plt.subplots(1, 1, figsize=(4, 6))
height_range = slice(6e3, 18e3)

for run in ["jed0011", "jed0022"]:
    mean_temp = datasets[run]["ta"].where(masks_clearsky[run]).mean("index")
    ax.plot(
        conv_real[run].where(
            (vgrid["zg"].sel(height_2=conv_real[run]["height_2"]) >= height_range.start)
            & (
                vgrid["zg"].sel(height_2=conv_real[run]["height_2"])
                <= height_range.stop
            )
        )
        * 86400,
        mean_temp.sel(height=conv_real[run]["height_2"]).where(
            (vgrid["zg"].sel(height_2=conv_real[run]["height_2"]) >= height_range.start)
            & (
                vgrid["zg"].sel(height_2=conv_real[run]["height_2"])
                <= height_range.stop
            )
        ),
        label=exp_name[run],
        color=colors[run],
    )
    ax.invert_yaxis()
    ax.set_ylim([260, 200])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("Temperature / K")
    ax.set_xlabel("Convergence / day$^{-1}$")


# %%
