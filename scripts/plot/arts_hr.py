# %%
import pyarts
import FluxSimulator as fsm
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
pyarts.cat.download.retrieve(verbose=True)


# %% load data
runs = ["jed0011", "jed0033", "jed0022"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
colors = {"jed0011": "k", "jed0022": "r", "jed0033": "orange"}
datasets = {}
hrs = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed.nc"
    ).sel(index=slice(0, 1e6))
    hrs[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_heating_rates.nc"
    ).sel(index=slice(0, 1e6))
vgrid = (
    xr.open_dataset(
        "/work/bm1183/m301049/icon-mpim/experiments/jed0001/atm_vgrid_angel.nc"
    )
    .mean("ncells")
    .rename({"height": "height_2", "height_2": "height"})
)

# %% calculate mean clearsky profiles
mean_profiles = {}
for run in runs:
    mask_clearsky = (datasets[run]["iwp"] < 1e-4) & (datasets[run]["lwp"] < 1e-4)
    mean_profiles[run] = (
        datasets[run][
            [
                "ta",
                "hus",
                "o3",
                "pfull",
                "rld",
                "rlu",
                "rsd",
                "rsu",
                "rho",
                "cli",
                "clw",
                "qs",
                "qg",
                "qr",
            ]
        ]
        .where(mask_clearsky)
        .mean("index")
    )
    mean_profiles[run] = mean_profiles[run].assign(
        {
            "net_hr": hrs[run]["net_hr"].where(mask_clearsky).mean("index"),
            "sw_hr": hrs[run]["sw_hr"].where(mask_clearsky).mean("index"),
            "lw_hr": hrs[run]["lw_hr"].where(mask_clearsky).mean("index"),
        }
    )

#  convert profiles to vmr
MMs = {"hus": 18, "o3": 48.00}  # g/mol


def convert_to_densiy(var, ds):
    var_dens = (var / ds["dry_air"]) * ds["rho_air"]
    return var_dens


def convert_to_vmr(rho_var, MM, rho_dry_air):
    vmr = (rho_var / rho_dry_air) * (28.964 / MM)
    return vmr


for run in runs:
    mean_profiles[run] = mean_profiles[run].assign(
        rho_air=lambda d: d["pfull"] / (287.04 * d["ta"])
    )
    mean_profiles[run] = mean_profiles[run].assign(
        dry_air=lambda d: 1 - (d.cli + d.clw + d.qs + d.qg + d.qr + d.hus)
    )
    mean_profiles[run]["rho_air"].attrs = {
        "units": "kg/m^3",
        "long_name": "density of dry air",
    }
    mean_profiles[run]["dry_air"].attrs = {
        "units": "kg/kg",
        "long_name": "specific mass of dry air",
    }
    for var in ["o3", "hus"]:
        mean_profiles[run] = mean_profiles[run].assign(
            var=convert_to_vmr(
                convert_to_densiy(mean_profiles[run][var], mean_profiles[run]),
                MMs[var],
                mean_profiles[run]["rho_air"],
            )
        )
        mean_profiles[run][var].attrs = {
            "units": "1",
            "long_name": f"volume mixing ratio of {var}",
        }


# %% plot mean profiles
fig, axes = plt.subplots(1, 4, figsize=(10, 8), sharey=True)

for run in runs:
    axes[0].plot(
        mean_profiles[run]["net_hr"],
        vgrid["zg"],
        color=colors[run],
    )
    axes[1].plot(
        mean_profiles[run]["ta"],
        vgrid["zg"],
        color=colors[run],
    )
    axes[2].plot(
        mean_profiles[run]["hus"],
        vgrid["zg"],
        color=colors[run],
    )
    axes[3].plot(
        mean_profiles[run]["o3"],
        vgrid["zg"],
        color=colors[run],
    )

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_xlabel("Net heating rate [K/day]")
axes[1].set_xlabel("Temperature [K]")
axes[2].set_xlabel("Humidity [VMR]")
axes[3].set_xlabel("Ozone [VMR]")


# %% convert xarray to ARTS gridded field
names = ["c-base", "2K-base", "4K-base", "2K-co3", "4K-co3", "2K-chus", "4K-chus"]
atms_grd = pyarts.arts.ArrayOfGriddedField4()

for run in runs:
    profile_grd = fsm.generate_gridded_field_from_profiles(
        mean_profiles[run]["pfull"].values.astype(float)[::-1],
        mean_profiles[run]["ta"].values.astype(float)[::-1],
        vgrid['zg'].values.astype(float)[::-1],
        gases={
            "H2O": mean_profiles[run]["hus"].values.astype(float)[::-1],
            "CO2": np.ones(len(mean_profiles["jed0011"]["hus"])) * 415e-6,
            "O3": mean_profiles[run]["o3"].values.astype(float)[::-1],
            "N2": np.ones(len(mean_profiles["jed0011"]["hus"])) * 0.7808,
            "O2": np.ones(len(mean_profiles["jed0011"]["hus"])) * 0.2095,
        },
    )
    atms_grd.append(profile_grd)

# %% setup flux simulator
gases = ["H2O", "CO2", "O2", "N2", "O3"]  # gases to include in the simulation
f_grid = np.linspace(1, 3e3, 200)  # frequency grid in cm^-1
f_grid_freq = pyarts.arts.convert.kaycm2freq(f_grid)  # frequency grid in Hz
surface_reflectivity_lw = 0.05  # surface reflectivity
setup_name = "aquaplanet"  # name of the simulation
species = [  # species to include in the simulation
    "H2O, H2O-SelfContCKDMT350, H2O-ForeignContCKDMT350",
    "O2-*-1e12-1e99,O2-CIAfunCKDMT100",
    "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
    "CO2, CO2-CKDMT252",
    "O3",
    "O3-XFIT",
]
LW_flux_simulator = fsm.FluxSimulator(setup_name)
LW_flux_simulator.ws.f_grid = f_grid_freq
LW_flux_simulator.set_species(species)

# %% get lookuptable
LW_flux_simulator.get_lookuptableBatch(atms_grd)

# %% calculate fluxes
results = {}
for idx, run in enumerate(runs):
    print(f"Calculating fluxes for {run}")
    results[run] = LW_flux_simulator.flux_simulator_single_profile(
        atms_grd[idx],
        float(mean_profiles[run]["ta"].isel(height=-1)),
        12.5,
        surface_reflectivity_lw,
    )

# %%
