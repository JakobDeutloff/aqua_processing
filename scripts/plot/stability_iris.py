# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# %% load data
runs = ["jed0011", "jed0022", "jed0033"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
datasets = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample.nc"
    ).sel(index=slice(0, 2e5))

vgrid = xr.open_dataset(
    "/work/bm1183/m301049/icon-mpim/experiments/jed0001/atm_vgrid_angel.nc"
).mean("ncells")
# %% get decrease of cloud fraction 
cf = {}
for run in runs:
    cf[run] = ((datasets[run]["clivi"] + datasets[run]["qsvi"] + datasets[run]["qgvi"]) > 1e-5).mean()


# %% calculate heating rates
def calc_heating_rates(ds):
    cp = 1004  # J kg^-1 K^-1 specific heat capacity of dry air at constant pressure
    sw_hr = (
        (1 / (ds["rho"] * cp))
        * (
            (ds["rsd"] - ds["rsu"]).diff("height")
            / (vgrid["zg"].diff("height_2").values)
        )
        * 86400
    )
    lw_hr = (
        (1 / (ds["rho"] * cp))
        * (
            (ds["rld"] - ds["rlu"]).diff("height")
            / (vgrid["zg"].diff("height_2").values)
        )
        * 86400
    )
    net_hr = sw_hr + lw_hr

    hr_arr = xr.Dataset(
        {
            "sw_hr": sw_hr,
            "lw_hr": lw_hr,
            "net_hr": net_hr,
        }
    )

    hr_arr["sw_hr"].attrs = {
        "units": "K/day",
        "long_name": "Shortwave heating rate",
    }
    hr_arr["lw_hr"].attrs = {
        "units": "K/day",
        "long_name": "Longwave heating rate",
    }
    hr_arr["net_hr"].attrs = {
        "units": "K/day",
        "long_name": "Net heating rate",
    }

    return hr_arr


def calc_stability(ds):

    cp = 1004  # J kg^-1 K^-1 specific heat capacity of dry air at constant pressure
    R = 287  # J kg^-1 K^-1 gas constant of dry air

    stab = (
        (ds["ta"] / ds["pfull"]) * (R / cp)
        - (ds["ta"].diff("height") / ds["pfull"].diff("height"))
    ) * 1e5

    stab.attrs = {
        "units": "mK hPa^-1",
        "long_name": "Stability",
    }

    return stab


def calc_w_sub(ds):
    wsub = (ds["net_hr"] / ds["stab"]) * 1e3
    wsub.attrs = {
        "units": "hPa day^-1",
        "long_name": "Subsidence velocity",
    }
    return -wsub


def calc_conv(wsub, pfull):
    conv = wsub.diff("height") / (pfull.diff("height") / 1e2)
    conv.attrs = {
        "units": "day^-1",
        "long_name": "Convergence",
    }
    return conv


def calc_cf(ds):
    cf = ((ds["clw"] + ds["qr"] + ds["cli"] + ds["qs"] + ds["qg"]) > 1e-5).astype(int)
    cf.attrs = {
        "units": "1",
        "long_name": "Cloud Mask",
    }
    return cf


# %% calculate instataneous values
hrs = {}
masks_clearsky = {}
for run in runs:
    print(run)
    hrs[run] = calc_heating_rates(datasets[run])
    hrs[run] = hrs[run].assign(stab=calc_stability(datasets[run]))
    datasets[run] = datasets[run].assign(cf=calc_cf(datasets[run]))

    masks_clearsky[run] = (
        datasets[run]["clivi"] + datasets[run]["qsvi"] + datasets[run]["qgvi"]
    ) < 1e-4

# %%   calculate mean values
subs = {}
conv = {}
idx_max_conv = {}
z_max_conv = {}
stab_max_conv = {}
cf_max_conv = {}
for run in runs:
    subs[run] = calc_w_sub(hrs[run].where(masks_clearsky[run]).mean("index"))
    conv[run] = calc_conv(
        subs[run], datasets[run]["pfull"].where(masks_clearsky[run]).mean("index")
    )
    idx_max_conv[run] = (
        conv[run]
        .where(
            (vgrid["zghalf"].sel(height=conv[run]["height"]) < 18e3)
            & (vgrid["zghalf"].sel(height=conv[run]["height"]) > 6e3)
        )
        .argmax("height")
    )
    z_max_conv[run] = vgrid["zghalf"].sel(height=idx_max_conv[run]).values
    stab_max_conv[run] = (
        hrs[run]["stab"]
        .where(masks_clearsky[run])
        .mean("index")
        .sel(height=idx_max_conv[run])
    )
    cf_max_conv[run] = datasets[run]["cf"].mean("index").sel(height=idx_max_conv[run])


# %% plot mean heating rate
fig, axes = plt.subplots(1, 6, figsize=(14, 6), sharey=True)
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
        subs[run].where(
            (vgrid["zghalf"].sel(height=subs[run]["height"]) >= height_range.start)
            & (vgrid["zghalf"].sel(height=subs[run]["height"]) <= height_range.stop)
        ),
        mean_temp.sel(height=subs[run]["height"]).where(
            (vgrid["zghalf"].sel(height=subs[run]["height"]) >= height_range.start)
            & (vgrid["zghalf"].sel(height=subs[run]["height"]) <= height_range.stop)
        ),
        label=exp_name[run],
        color=colors[run],
    )
    axes[3].plot(
        conv[run].where(
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
    axes[5].plot(
        datasets[run]["pfull"]
        .where(masks_clearsky[run])
        .mean("index")
        .where(
            (vgrid["zghalf"].sel(height=datasets[run]["height"]) >= height_range.start)
            & (vgrid["zghalf"].sel(height=datasets[run]["height"]) <= height_range.stop)
        )
        .diff("height")
        / 1e2,
        mean_temp.sel(height=hrs[run]["height"]).where(
            (vgrid["zghalf"].sel(height=hrs[run]["height"]) >= height_range.start)
            & (vgrid["zghalf"].sel(height=hrs[run]["height"]) <= height_range.stop)
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


# %% scatter plots of values at max_conv vs Ts
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
surftemps = {"jed0011": 0, "jed0022": 4, "jed0033": 2}

for run in runs:
    axes[0].scatter(
        surftemps[run],
        stab_max_conv[run],
        label=exp_name[run],
        color=colors[run],
    )
    axes[1].scatter(
        surftemps[run],
        z_max_conv[run] / 1e3,
        label=exp_name[run],
        color=colors[run],
    )
    axes[2].scatter(
        surftemps[run],
        cf_max_conv[run],
        label=exp_name[run],
        color=colors[run],
    )
for ax in axes:
    ax.set_xlabel(r"$T_{\mathrm{s}}$ anomaly / K")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([0, 2, 4])


axes[0].set_ylabel("Stability at max convergence / mK hPa$^{-1}$")
axes[1].set_ylabel("Height of max convergence / km")
axes[2].set_ylabel("Cloud Fraction at max convergence")

fig.tight_layout()

# %% new calculateions based on jevanjee


def calc_stability_jev(ds):
    g = -9.81
    cp = 1004
    stab = (g / cp) - (ds["ta"].diff("height") / vgrid["zg"].diff("height_2").values)
    stab.attrs = {
        "units": "K m^-1",
        "long_name": "Stability",
    }
    return stab


def calc_w_sub_jev(ds):
    wsub = ds["net_hr"] / ds["stab"]
    wsub.attrs = {
        "units": "m day^-1",
        "long_name": "Subsidence velocity",
    }
    return wsub


def calc_conv_jev(wsub):
    conv = wsub.diff("height") / (
        vgrid["zghalf"].sel(height=wsub["height"]).diff("height").values
    )
    conv.attrs = {
        "units": "day^-1",
        "long_name": "Convergence",
    }
    return conv


# %% calculte jevanjee values
hrs_jev = {}
masks_clearsky = {}
for run in runs:
    print(run)
    hrs_jev[run] = calc_heating_rates(datasets[run])
    hrs_jev[run] = hrs_jev[run].assign(stab=calc_stability_jev(datasets[run]))
    datasets[run] = datasets[run].assign(cf=calc_cf(datasets[run]))

    masks_clearsky[run] = (
        datasets[run]["clivi"] + datasets[run]["qsvi"] + datasets[run]["qgvi"]
    ) < 1e-4

subs_jev = {}
conv_jev = {}
for run in runs:
    subs_jev[run] = calc_w_sub_jev(
        hrs_jev[run].where(masks_clearsky[run]).median("index")
    )
    conv_jev[run] = calc_conv_jev(subs_jev[run])


# %% plot results jevanjee

fig, axes = plt.subplots(1, 4, figsize=(14, 6), sharey=True)

for run in runs:
    mean_temp = datasets[run]["ta"].where(masks_clearsky[run]).mean("index")
    axes[0].plot(
        hrs_jev[run]["net_hr"]
        .where(masks_clearsky[run])
        .median("index")
        .where(
            (vgrid["zghalf"].sel(height=hrs_jev[run]["height"]) >= height_range.start)
            & (vgrid["zghalf"].sel(height=hrs_jev[run]["height"]) <= height_range.stop)
        ),
        mean_temp.sel(height=hrs_jev[run]["height"]).where(
            (vgrid["zghalf"].sel(height=hrs_jev[run]["height"]) >= height_range.start)
            & (vgrid["zghalf"].sel(height=hrs_jev[run]["height"]) <= height_range.stop)
        ),
        label=exp_name[run],
        color=colors[run],
    )
    axes[1].plot(
        hrs_jev[run]["stab"]
        .where(masks_clearsky[run])
        .median("index")
        .where(
            (vgrid["zghalf"].sel(height=hrs_jev[run]["height"]) >= height_range.start)
            & (vgrid["zghalf"].sel(height=hrs_jev[run]["height"]) <= height_range.stop)
        ),
        mean_temp.sel(height=hrs_jev[run]["height"]).where(
            (vgrid["zghalf"].sel(height=hrs_jev[run]["height"]) >= height_range.start)
            & (vgrid["zghalf"].sel(height=hrs_jev[run]["height"]) <= height_range.stop)
        ),
        label=exp_name[run],
        color=colors[run],
    )
    axes[2].plot(
        subs_jev[run].where(
            (vgrid["zghalf"].sel(height=subs_jev[run]["height"]) >= height_range.start)
            & (vgrid["zghalf"].sel(height=subs_jev[run]["height"]) <= height_range.stop)
        ),
        mean_temp.sel(height=subs_jev[run]["height"]).where(
            (vgrid["zghalf"].sel(height=subs_jev[run]["height"]) >= height_range.start)
        ),
        label=exp_name[run],
        color=colors[run],
    )
    axes[3].plot(
        -conv_jev[run].where(
            (vgrid["zghalf"].sel(height=conv_jev[run]["height"]) >= height_range.start)
            & (vgrid["zghalf"].sel(height=conv_jev[run]["height"]) <= height_range.stop)
        ),
        mean_temp.sel(height=conv_jev[run]["height"]).where(
            (vgrid["zghalf"].sel(height=conv_jev[run]["height"]) >= height_range.start)
            & (vgrid["zghalf"].sel(height=conv_jev[run]["height"]) <= height_range.stop)
        ),
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
axes[1].set_xlim([-0.01, 0])
axes[0].invert_yaxis()
axes[0].set_ylim([260, 200])
fig.tight_layout()
fig.savefig('plots/iwp_drivers/stab_iris.png', dpi=300)

# %% compare actual lapse rates to moist adiabats
laps_rate = {}
masks_clearsky = {}
for run in runs:
    masks_clearsky[run] = (
        datasets[run]["clivi"] + datasets[run]["qsvi"] + datasets[run]["qgvi"]
    ) < 1e-6
    laps_rate[run] = np.abs((
        (datasets[run]["ta"].diff("height") / vgrid["zg"].diff("height_2").values)
        .where(masks_clearsky[run])
        .mean("index")
    ))


# %% plot lapse rates
fig, axes = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
height_range = slice(10e3, 18e3)

for run in runs:
    mean_temp = datasets[run]["ta"].where(masks_clearsky[run]).mean("index")
    axes[0].plot(
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
    axes[1].plot(
        laps_rate[run].where(
            (vgrid["zghalf"].sel(height=laps_rate[run]["height"]) >= height_range.start)
            & (
                vgrid["zghalf"].sel(height=laps_rate[run]["height"])
                <= height_range.stop
            )
        ) - 
        laps_rate["jed0011"].where(
            (vgrid["zghalf"].sel(height=laps_rate[run]["height"]) >= height_range.start)
            & (
                vgrid["zghalf"].sel(height=laps_rate[run]["height"])
                <= height_range.stop
            )
        ),
        mean_temp.sel(height=laps_rate[run]["height"]).where(
            (vgrid["zghalf"].sel(height=datasets[run]["height"]) >= height_range.start)
            & (vgrid["zghalf"].sel(height=datasets[run]["height"]) <= height_range.stop)
        ),
        label=exp_name[run],
        color=colors[run],
    )

axes[0].invert_yaxis()

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
height_range = slice(10e3, 18e3)

for run in ["jed0011", "jed0022"]:
    ax.plot(
        conv_real[run].where(
            (vgrid["zg"].sel(height_2=conv_real[run]["height_2"]) >= height_range.start)
            & (vgrid["zg"].sel(height_2=conv_real[run]["height_2"]) <= height_range.stop)
        ),
        vgrid["zg"].sel(height_2=conv_real[run]["height_2"]).where(
            (vgrid["zg"].sel(height_2=conv_real[run]["height_2"]) >= height_range.start)
            & (vgrid["zg"].sel(height_2=conv_real[run]["height_2"]) <= height_range.stop)
        ),
        label=exp_name[run],
        color=colors[run],
    )



# %%
