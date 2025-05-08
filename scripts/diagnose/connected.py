# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# %% load data
runs = ["jed0011", "jed0033", "jed0022"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
colors = {"jed0011": "k", "jed0022": "r", "jed0033": "orange"}
datasets = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed.nc"
    ).sel(index=slice(0, 1e6))

vgrid = (
    xr.open_dataset(
        "/work/bm1183/m301049/icon-mpim/experiments/jed0001/atm_vgrid_angel.nc"
    )
    .mean("ncells")
    .rename({"height_2": "height", "height": "height_2"})
)


# %% convert to density
def convert_to_densiy(var, ds):
    var_dens = (var / ds["dry_air"]) * ds["rho_air"]
    return var_dens


datasets_dens = {}
hydrometeors = ["cli", "clw", "qs", "qg", "qr"]
for run in runs:
    datasets[run] = datasets[run].assign(
        rho_air=lambda d: d["pfull"] / (287.04 * d["ta"])
    )
    datasets[run] = datasets[run].assign(
        dry_air=lambda d: 1 - (d.cli + d.clw + d.qs + d.qg + d.qr + d.hus)
    )
for run in runs:
    datasets_dens[run] = datasets[run][["iwp", "lwp", "mask_height"]].copy()
    for var in hydrometeors:
        datasets_dens[run][var] = convert_to_densiy(datasets[run][var], datasets[run])
        datasets_dens[run][var].attrs = {
            "long_name": f"{var} density",
            "units": "kg m^-3",
        }


# %%
def calc_connected(ds, zg, frac_no_cloud=0.05, mean_height=11645):
    """
    defines for all profiles with ice above liquid whether
    the high and low clouds are connected (1) or not (0).
    Profiles where not both cloud types are present are filled with nan.
    Profiles masked aout in atm will also be nan.

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing atmospheric profiles, can be masked if needed
    frac_no_cloud : float
        Fraction of maximum cloud condensate in column to define no cloud

    Returns:
    --------
    connected : xarray.DataArray
        DataArray containing connectedness for each profile
    """

    # define liquid and ice cloud condensate
    liq = ds["clw"] + ds["qr"]
    ice = ds["cli"] + ds["qs"] + ds["qg"]
    cloud_top = zg.sel(height=ice.argmax("height"))
    cloud_bottom = zg.sel(height=liq.argmax("height"))
    cloud_height = (cloud_top - cloud_bottom).mean()

    # define ice and liquid content needed for connectedness
    no_ice_cloud = (ice > (frac_no_cloud * (mean_height/cloud_height)**12 * ice.max("height"))) * 1
    no_liq_cloud = (liq > (frac_no_cloud * (mean_height/cloud_height)**12 * liq.max("height"))) * 1
    no_cld = (
        no_liq_cloud + no_ice_cloud
    ) - 1  # -1 for cells with not enough condensate

    # find all profiles with ice above liquid
    mask_both_clds = (ds["lwp"] > 1e-4) & (ds["iwp"] > 1e-4)

    # create connectedness array
    connected = xr.DataArray(
        np.ones(mask_both_clds.shape),
        coords=mask_both_clds.coords,
        dims=mask_both_clds.dims,
    )
    connected.attrs = {
        "units": "1",
        "long_name": "Connectedness of liquid and frozen clouds",
    }

    # set all profiles with no liquid cloud to nan
    connected = connected.where(mask_both_clds, np.nan)

    # Calculate the height of maximum ice and liquid content
    h_ice = ice.argmax("height")
    h_liq = liq.argmax("height")

    # Calculate the mean cloud content between the heights of maximum ice and liquid content
    cld_range_mean = no_cld.where(
        (no_cld.height >= h_ice) & (no_cld.height <= h_liq)
    ).sum("height")

    # Determine connectedness
    connected = xr.where(cld_range_mean < -1, 0, connected)

    return connected


# %% calculate connectedness
for run in runs:
    conn = calc_connected(datasets[run], vgrid["zg"], frac_no_cloud=0.05)
    datasets_dens[run] = datasets_dens[run].assign(conn=conn)

# %% calculate lc fraction
iwp_bins = np.logspace(-4, 1, 51)
iwp_points = (iwp_bins[:-1] + iwp_bins[1:]) / 2
lc_fraction = {}
liqc_fraction = {}
conn_liquid = {}
for run in runs:
    lc_fraction[run] = (
        ((datasets_dens[run]["lwp"] > 1e-4) & (datasets_dens[run]['conn'] == 0))
        .groupby_bins(datasets_dens[run]["iwp"], iwp_bins, labels=iwp_points)
        .mean()
    )
    liqc_fraction[run] = (
        (datasets_dens[run]["lwp"] > 1e-4)
        .groupby_bins(datasets_dens[run]["iwp"], iwp_bins, labels=iwp_points)
        .mean()
    )
    conn_liquid[run] = (
        (datasets_dens[run]["conn"])
        .groupby_bins(datasets_dens[run]["iwp"], iwp_bins, labels=iwp_points)
        .mean()
    )

# %% plot low cloud fraction
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

for run in runs:
    ax.plot(iwp_points, lc_fraction[run], label=exp_name[run], color=colors[run])

ax.set_xscale("log")
# %% plot liquid cloud fraction
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for run in runs:
    ax.plot(iwp_points, liqc_fraction[run], label=exp_name[run], color=colors[run])

ax.set_xscale("log")

# %% plot share of connected liquid clouds
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for run in runs:
    ax.plot(
        iwp_points, conn_liquid[run], label=exp_name[run], color=colors[run]
    )
ax.set_xscale("log")
ax.set_yscale("log")
# %% calculate mean profiles
iwp_bins_coarse = np.logspace(-4, 1, 6)
# Define the variables to process
variables = {
    "Liquid": ["clw", "qr"],
    "Frozen": ["cli", "qs", "qg"],
    "Snow": ["qs"],
    "Rain": ["qr"],
    "Graupel": ["qg"],
    "Cloud Ice": ["cli"],
    "Cloud Water": ["clw"],
    "conn": ["conn"],
}

# Initialize dictionaries for each variable
results = {var: {} for var in variables}

# Process each run and variable
for run in runs:
    for var, components in variables.items():
        results[var][run] = (
            sum(datasets_dens[run][comp] for comp in components)
            .groupby_bins(datasets_dens[run]["iwp"], iwp_bins_coarse)
            .mean()
        )
# %% plot mean profiles
fig, axes = plt.subplots(3, 5, figsize=(15, 15), sharey=True, sharex="col")
axtwin = np.array([ax.twiny() for ax in axes.flatten()]).reshape(axes.shape)
linstyles = ["-", "--", "-.", ":"]

# Define the variables to plot
liq_vars = ["Liquid", "Cloud Water", "Rain"]
ice_vars = ["Frozen", "Cloud Ice", "Snow", "Graupel"]

for i, run in enumerate(runs):
    for j in range(len(iwp_bins_coarse) - 1):
        # Plot primary variables on the main axes
        for k, var in enumerate(liq_vars):
            axes[i, j].plot(
                results[var][run].isel(iwp_bins=j),
                vgrid["zg"] / 1e3,
                label=var,
                color="k",
                linestyle=linstyles[k],
            )

        axes[i, j].text(
            0.8,
            0.9,
            f"{float((results['conn'][run].isel(iwp_bins=j).values * 100).round(2))}% ",
            transform=axes[i, j].transAxes,
        )
        # Plot secondary variables on the twin axes
        for k, var in enumerate(ice_vars):
            axtwin[i, j].plot(
                results[var][run].isel(iwp_bins=j),
                vgrid["zg"] / 1e3,
                label=var,
                color="b",
                linestyle=linstyles[k % len(linstyles)],
            )

# Set y-axis limits for all axes
for ax in axes.flatten():
    ax.set_ylim(0, 18)


# Ensure all twin axes in the same column share their x-axis
for i in range(axtwin.shape[1]):
    for j in range(axtwin.shape[0]):
        axtwin[j, i].sharex(axtwin[0, i])

for i, ax in enumerate(axes[-1, :]):
    ax.text(
        0,
        -0.2,
        f"IWP Bin: {iwp_bins_coarse[i]} - {iwp_bins_coarse[i + 1]} kg m$^{-2}$",
        transform=ax.transAxes,
        fontweight="bold",
    )

for i, ax in enumerate(axes[:, 0]):
    ax.set_ylabel("Height / km")
    ax.text(
        -0.3,
        0.5,
        exp_name[runs[i]],
        transform=ax.transAxes,
        rotation=90,
        verticalalignment="center",
        fontweight="bold",
    )

for ax in axes[-1, :]:
    ax.set_xlabel("Liquid / kg m$^{-3}$")

for ax in axtwin[0, :]:
    ax.set_xlabel("Ice / kg m$^{-3}$", color="b")

handles_liq, labels_liq = axes[0, 0].get_legend_handles_labels()
handles_ice, labels_ice = axtwin[0, 0].get_legend_handles_labels()
handles = handles_liq + handles_ice
labels = labels_liq + labels_ice
fig.legend(handles, labels, bbox_to_anchor=(0.8, 0), ncols=7)

fig.tight_layout()
fig.savefig(
    "plots/misc/mean_profiles_connectedness.png",
    dpi=300,
    bbox_inches="tight",
)


# %%
