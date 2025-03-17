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

# %% load data
runs = ["jed0011", "jed0022", "jed0033"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
datasets = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample.nc"
    ).sel(index=slice(0, 1e5))
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
hrs_jev = {}
for run in runs:
    print(run)
    hrs_jev[run] = calc_heating_rates(
        datasets[run]["rho"].where(mask_trop[run]),
        datasets[run]["rsd"].where(mask_trop[run])
        - datasets[run]["rsu"].where(mask_trop[run]),
        datasets[run]["rld"].where(mask_trop[run])
        - datasets[run]["rlu"].where(mask_trop[run]),
        vgrid,
    )
    hrs_jev[run] = hrs_jev[run].assign(
        stab=calc_stability(datasets[run]["ta"].where(mask_trop[run]), vgrid=vgrid)
    )
#  calcualte sub and conv from mean values
sub_mean = {}
conv_mean = {}
for run in runs:
    mean_hrs = hrs_jev[run].where(masks_clearsky[run]).mean("index")
    sub_mean[run] = calc_w_sub(mean_hrs["net_hr"], mean_hrs["stab"])
    conv_mean[run] = calc_conv(sub_mean[run], vgrid)


# %% plot results jevanjee
height_range = slice(6e3, 16e3)
colors = {"jed0011": "k", "jed0022": "r", "jed0033": "orange"}
fig, axes = plt.subplots(1, 4, figsize=(14, 6), sharey=True)
mask_hrs = (
    vgrid["zghalf"].sel(height_2=hrs_jev["jed0011"]["height"]) >= height_range.start
) & (vgrid["zghalf"].sel(height_2=hrs_jev["jed0011"]["height"]) <= height_range.stop)
mask_conv = (
    vgrid["zghalf"].sel(height_2=conv_mean["jed0011"]["height"]) >= height_range.start
) & (vgrid["zghalf"].sel(height_2=conv_mean["jed0011"]["height"]) <= height_range.stop)


for run in runs:
    mean_temp = (
        datasets[run]["ta"].where(mask_trop[run] & masks_clearsky[run]).mean("index")
    )
    axes[0].plot(
        hrs_jev[run]["net_hr"].where(masks_clearsky[run]).mean("index").where(mask_hrs),
        mean_temp.sel(height=hrs_jev[run]["height"]).where(mask_hrs),
        label=exp_name[run],
        color=colors[run],
    )
    axes[1].plot(
        hrs_jev[run]["stab"].where(masks_clearsky[run]).mean("index").where(mask_hrs),
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


for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)


axes[0].set_ylabel("Temperature / K")
axes[0].set_xlabel("Heating rate / K day$^{-1}$")
axes[1].set_xlabel("Stability / K m$^{-1}$")
axes[2].set_xlabel("Subsidence / m day$^{-1}$")
axes[3].set_xlabel("Convergence / day$^{-1}$")
axes[0].set_xlim([-1.5, 0.3])
axes[1].set_xlim([-0.006, -0.001])
axes[0].invert_yaxis()
axes[0].set_ylim([260, 200])
fig.tight_layout()
fig.savefig("plots/iwp_drivers/stab_iris.png", dpi=300)

# %% compare actual lapse rates
laps_rate = {}
for run in runs:
    laps_rate[run] = np.abs(
        (
            (datasets[run]["ta"].diff("height") / vgrid["zg"].diff("height").values)
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
            / vgrid["zghalf"][:-1].diff("height_2").values
        )
        .where(masks_clearsky[run] & mask_trop[run].rename({"height": "height_2"}))
        .mean("index")
    )

mask_conv_real = (
    vgrid["zghalf"].sel(height_2=conv_real["jed0011"]['height_2']) >= height_range.start
) & (vgrid["zghalf"].sel(height_2=conv_real["jed0011"]['height_2']) <= height_range.stop)

fig, ax = plt.subplots(1, 1, figsize=(4, 6))

for run in ["jed0011", "jed0022"]:
    mean_temp = (
        datasets[run]["ta"].where(masks_clearsky[run] & mask_trop[run]).mean("index")
    )
    ax.plot(
        conv_real[run].where(mask_conv_real) * 86400,
        mean_temp.sel(height=conv_real[run]["height_2"]).where(mask_conv_real),
        label=exp_name[run],
        color=colors[run],
    )
    ax.invert_yaxis()
    ax.set_ylim([260, 200])
    ax.set_xlim([-0.2, 0.3])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("Temperature / K")
    ax.set_xlabel("Convergence / day$^{-1}$")


