# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import datetime as dt
from concurrent.futures import ProcessPoolExecutor
from src.grid_helpers import merge_grid, fix_time
from src.moist_layers import plot_prw_field, get_contour_length
from src.read_data import load_random_datasets, load_vgrid
from dask.diagnostics import ProgressBar
from scipy.stats import linregress
from functools import partial
import pandas as pd

# %% load data
ds = xr.open_dataset(
    "/work/bm1183/m301049/icon_hcap_data/plus2K/production/latlon/atm2d_latlon.nc"
)
ds_rand = load_random_datasets()["jed0033"]
ds_north = xr.open_dataset(
    "/work/bm1183/m301049/icon_hcap_data/plus2K/production/latlon/zonal_north_15.nc"
)
ds_north = ds_north.sortby("time")
ds_south = xr.open_dataset(
    "/work/bm1183/m301049/icon_hcap_data/plus2K/production/latlon/zonal_south_15.nc"
)
ds_south = ds_south.sortby("time")
vgrid = load_vgrid()
# %% check histogram
fig, ax = plt.subplots(figsize=(8, 5))
ds["prw"].isel(time=slice(80, 90)).sel(lat=slice(-40, 40)).plot.hist(
    bins=np.arange(0, 60, 1),
    ax=ax,
    color="grey",
)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig("plots/moist_layers/prw_histogram_40_40.png", dpi=300, bbox_inches="tight")

# %% plot pr vs prw
ds_2d = (
    xr.open_dataset(
        "/work/bm1183/m301049/icon-mpim/experiments/jed0011/jed0011_atm_2d_19790729T000040Z.15371960.nc"
    )
    .pipe(merge_grid)
    .pipe(fix_time)
)

fig, ax = plt.subplots(figsize=(8, 5))
prw_bins = np.arange(0, 71, 1)
binned_pr = (
    (ds_2d["pr"] * 60 * 60 * 24)
    .isel(time=0)
    .where((ds_2d["clat"] < 20) & (ds_2d["clat"] > -20))
    .groupby_bins(ds_2d["prw"].isel(time=0), bins=prw_bins)
)
binned_pr.mean().plot(ax=ax, color="k", label="Mean PR")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("Water Vapor Path / mm$")
ax.set_ylabel("Precipitation / mm day$^{-1}$")
ax.set_title("")
ax.set_yscale("log")
ax.set_ylim(0, 100)
fig.savefig("plots/moist_layers/pr_vs_prw_20_20.png", dpi=300, bbox_inches="tight")


# %% calculate contourlength
def contour_length_for_time(t, cont=31):
    # Helper for parallel execution
    return int(get_contour_length(ds["prw"].sel(time=t), cont))


with ProcessPoolExecutor(max_workers=64) as executor:
    contour_lenghts = list(
        tqdm(
            executor.map(contour_length_for_time, ds["time"].values),
            total=len(ds["time"]),
        )
    )

contour_lenghts_30 = xr.DataArray(
    contour_lenghts / np.max(contour_lenghts),
    dims=["time"],
    coords={"time": ds["prw"].time},
    attrs={"long_name": "Contour length of 31 mm PRW"},
)

# %% plot contour length
fig, ax = plt.subplots(figsize=(10, 5))
contour_lenghts_30.plot(ax=ax, color="k", label="Contour length")
ax.set_xlabel("Time")
ax.set_ylabel("Normalized 31 mm contour length")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig("plots/moist_layers/contour_length_31.png", dpi=300, bbox_inches="tight")


# %% plot max and min organisation
max_idx = contour_lenghts_30.argmax().item()
min_idx = contour_lenghts_30.argmin().item()
fig, ax = plot_prw_field(ds["prw"].isel(time=max_idx), 32)
fig.savefig("plots/moist_layers/prw_max_31mm.png", dpi=300, bbox_inches="tight")
fig, ax = plot_prw_field(ds["prw"].isel(time=min_idx), 32)
fig.savefig("plots/moist_layers/prw_min_31mm.png", dpi=300, bbox_inches="tight")

# %% calculate mse
mse = ds_rand["ta"] * 1004 + ds_rand["hus"] * 2.5e6 + 9.81 * vgrid["zg"]


# %% transport
def calculate_variances(mask, v, q, mse):
    """Helper function to calculate variances and covariance."""
    mean_v = v.where(mask).mean("index")
    mean_q = q.where(mask).mean("index")
    mean_mse = mse.where(mask).mean("index")
    v_var = np.abs(v.where(mask) - mean_v).groupby(v.time.dt.date).mean()
    q_var = np.abs(q.where(mask) - mean_q).groupby(v.time.dt.date).mean()
    mse_var = np.abs(mse.where(mask) - mean_mse).groupby(mse.time.dt.date).mean()
    cov = (
        ((v.where(mask) - mean_v) * (q.where(mask) - mean_q))
        .groupby(v.time.dt.date)
        .mean()
    )
    cov_mse = (
        ((mse.where(mask) - mean_mse) * (q.where(mask) - mean_q))
        .groupby(mse.time.dt.date)
        .mean()
    )

    return mean_v, mean_q, mean_mse, v_var, q_var, mse_var, cov, cov_mse


# Dataset variables
v = ds_rand["va"]
q = ds_rand["hus"] * ds_rand["rho"]
v_adjusted = xr.where(ds_rand["clat"] < 0, -1 * v, v)

mask = (ds_rand["clat"] > -15) & (ds_rand["clat"] < 15)
mean_v, mean_q, mean_mse, v_var, q_var, mse_var, cov, cov_mse = calculate_variances(
    mask, v_adjusted, q, mse
)

# %% integrate fluxes over height
r_earth = 6371e3
len_tropics = r_earth * np.cos(np.deg2rad(15)) * 2 * np.pi * 2
dz = -vgrid["zghalf"].diff("height_2").values
eddy_moisture_export = (cov * dz).sum("height") * len_tropics
eddy_mse_export = (cov_mse * dz).sum("height") * len_tropics
mean_moisture_export = (mean_v * mean_q * dz).sum("height") * len_tropics
mean_mse_export = (mean_v * mean_mse * dz).sum("height") * len_tropics

# %% plot mean moisture export profile and eddy profiles
fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
axes[0].plot(mean_v * mean_q, vgrid["zg"] / 1e3, color="k", label="Mean")
axes[1].plot(cov.mean("date"), vgrid["zg"] / 1e3, color="k", label="Eddy")
axes[1].fill_betweenx(
    vgrid["zg"] / 1e3,
    cov.mean("date") - cov.std("date"),
    cov.mean("date") + cov.std("date"),
    color="grey",
    alpha=0.5,
    label="1 std",
)
for ax in axes:
    ax.set_ylim(0, 16)

    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_xlabel(r"$\overline{v} \cdot \overline{q}$ / kg s$^{-1}$")
axes[1].set_xlabel(r"$\overline{v' \cdot q'}$  / kg s$^{-1}$")
fig.tight_layout()
fig.savefig(
    "plots/moist_layers/moisture_export_profiles.png", dpi=300, bbox_inches="tight"
)

# %% plot mean mse export profile and eddy profiles
fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
axes[0].plot(mean_v * mean_mse, vgrid["zg"] / 1e3, color="k", label="Mean")
axes[1].plot(cov_mse.mean("date"), vgrid["zg"] / 1e3, color="k", label="Eddy")
axes[1].fill_betweenx(
    vgrid["zg"] / 1e3,
    cov_mse.mean("date") - cov_mse.std("date"),
    cov_mse.mean("date") + cov_mse.std("date"),
    color="grey",
    alpha=0.5,
    label="1 std",
)
for ax in axes:
    ax.set_ylim(0, 16)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_xlabel(r"$\overline{v} \cdot \overline{h}$ / W")
axes[1].set_xlabel(r"$\overline{v' \cdot h'}$  / W ")
fig.tight_layout()
fig.savefig("plots/moist_layers/mse_export_profiles.png", dpi=300, bbox_inches="tight")
# %% correlate eddy moisture export and contour length

slope, intercept, r_value, p_value, std_err = linregress(
    contour_lenghts_30, eddy_moisture_export
)
x = np.linspace(contour_lenghts_30.min().item(), contour_lenghts_30.max().item(), 10)
y = intercept + slope * x
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(contour_lenghts_30, eddy_moisture_export, color="k", s=5, alpha=0.5)
ax.plot(x, y, color="C1", label=f"r={r_value:.2f}")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("Normalized 32 mm contour length")
ax.set_ylabel("Eddy Moisture Export / kg s$^{-1}$")
fig.savefig(
    "plots/moist_layers/eddy_moisture_export_vs_contour_length.png",
    dpi=300,
    bbox_inches="tight",
)

# %% correlate eddy mse export and contour length
slope, intercept, r_value, p_value, std_err = linregress(
    contour_lenghts_30, eddy_mse_export
)
x = np.linspace(contour_lenghts_30.min().item(), contour_lenghts_30.max().item(), 10)
y = intercept + slope * x
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(contour_lenghts_30, eddy_mse_export, color="k", s=5, alpha=0.5)
ax.plot(x, y, color="C1", label=f"r={r_value:.2f}")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("Normalized 32 mm contour length")
ax.set_ylabel("Eddy MSE Export / W")
fig.savefig(
    "plots/moist_layers/eddy_mse_export_vs_contour_length.png",
    dpi=300,
    bbox_inches="tight",
)

# %% calculate energy fluxed from interpolated fluxes
mse_north = ds_north["ta"] * 1004 + ds_north["hus"] * 2.5e6 + 9.81 * vgrid["zg"]
mse_south = ds_south["ta"] * 1004 + ds_south["hus"] * 2.5e6 + 9.81 * vgrid["zg"]

# calculate mean and anomalies
mse = 0.5 * (mse_north + mse_south)
v = 0.5 * (ds_north["va"] - ds_south["va"])


# integrate
r_earth = 6371e3
dz = -vgrid["zghalf"].diff("height_2").values
width = r_earth * np.deg2rad(0.1)

mse_flux_north = (
    (mse_north * ds_north["va"] * ds_north["rho"] * dz).sum("height") * width
).sum("clon_bins")
mse_flux_south = (
    (mse_south * ds_south["va"] * ds_north["rho"] * -1 * dz).sum("height") * width
).sum("clon_bins")
total_mse_flux = mse_flux_north + mse_flux_south

# %% plot profiles of mse and v
fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
axes[0].plot(v.mean(["time", "clon_bins"]), vgrid["zg"] / 1e3, color="k", label="North")
axes[0].fill_betweenx(
    vgrid["zg"] / 1e3,
    v.mean(["time", "clon_bins"]) - v.std(["time", "clon_bins"]),
    v.mean(["time", "clon_bins"]) + v.std(["time", "clon_bins"]),
    color="k",
    alpha=0.5,
    label="1 std",
)
axes[1].plot(
    mse.mean(["time", "clon_bins"]) / 1e3, vgrid["zg"] / 1e3, color="k", label="North"
)
axes[1].fill_betweenx(
    vgrid["zg"] / 1e3,
    (mse).mean(["time", "clon_bins"]) / 1e3 - mse.std(["time", "clon_bins"]) / 1e3,
    mse.mean(["time", "clon_bins"]) / 1e3 + mse.std(["time", "clon_bins"]) / 1e3,
    color="k",
    alpha=0.5,
    label="1 std",
)

axes[0].set_ylim(0, 16)
axes[1].set_xlim(3.19e2, 3.5e2)
axes[0].set_xlim(-9, 9)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_xlabel(r"v / m s$^{-1}$")
axes[1].set_xlabel(r"h / kJ kg$^{-1}$")
axes[0].set_ylabel("Height / km")
fig.tight_layout()
fig.savefig("plots/moist_layers/v_mse_profiles.png", dpi=300, bbox_inches="tight")


# %% correlate contour length with eddy mse flux
e = (total_mse_flux - total_mse_flux.mean())/1e15
slope, intercept, r_value, p_value, std_err = linregress(contour_lenghts_30, e)
x = np.linspace(contour_lenghts_30.min().item(), contour_lenghts_30.max().item(), 10)
y = intercept + slope * x
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(contour_lenghts_30, e, color="k", s=5, alpha=0.5)
ax.plot(x, y, color="C1", label=f"r={r_value:.2f}")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("Normalized 31 mm contour length")
ax.set_ylabel("Eddy MSE Flux / PW")
fig.savefig(
    "plots/moist_layers/eddy_mse_flux_vs_contour_length.png",
    dpi=300,
    bbox_inches="tight",
)

# %% plot total mse flux
fig, ax = plt.subplots(figsize=(10, 5))
(total_mse_flux / 1e15).plot(ax=ax, color="k", label="Total MSE Flux")
ax.axhline(total_mse_flux.mean() / 1e15, color="orange", linestyle="--", label="Mean")
ax.set_xlabel("Time")
ax.set_ylabel("Total MSE Flux / PW")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(
    "plots/moist_layers/total_mse_flux.png", dpi=300, bbox_inches="tight"
)

# %% plot water path field at max and min energy transport
max_idx = total_mse_flux.argmax().item()
min_idx = total_mse_flux.argmin().item()
fig, ax = plot_prw_field(ds["prw"].isel(time=max_idx), 31)
fig.savefig(
    "plots/moist_layers/prw_max_mse_flux_32mm.png", dpi=300, bbox_inches="tight"
)
fig, ax = plot_prw_field(ds["prw"].isel(time=min_idx), 31)
fig.savefig(
    "plots/moist_layers/prw_min_mse_flux_32mm.png", dpi=300, bbox_inches="tight"
)

# %% calculate series of contour line lenghts 
thresholds = np.arange(24, 48, 1)

def contour_length_for_time(t, cont):
    # Helper for parallel execution
    return int(get_contour_length(ds["prw"].sel(time=t), cont=cont))

def get_c_lenghts(cont):

    with ProcessPoolExecutor(max_workers=64) as executor:
        func = partial(contour_length_for_time, cont=cont)
        contour_lenghts = list(
            tqdm(
                executor.map(func, ds["time"].values),
                total=len(ds["time"]),
            )
        )

    contour_lenghts = xr.DataArray(
        contour_lenghts / np.max(contour_lenghts),
        dims=["time"],
        coords={"time": ds["prw"].time},
        attrs={"long_name": f"Contour length of {cont} mm PRW"},
    )
    return contour_lenghts

c_lentghts = {}
for cont in thresholds:
    c_lentghts[str(cont)] = get_c_lenghts(cont)
# %% calculate correlation of all contour lengths with eddy mse flux
correlations = {}
for cont in thresholds:
    corr = xr.corr(c_lentghts[str(cont)], e)
    correlations[str(cont)] = corr.item()
correlations = pd.DataFrame.from_dict(correlations, orient="index", columns=["r"])
correlations.index = correlations.index.astype(int)

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(correlations.index, correlations["r"], color="k")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("Contour Line / mm")
ax.set_ylabel("Correlation with Eddy MSE Flux")
fig.savefig("plots/moist_layers/correlation_contour_length.png", dpi=300, bbox_inches="tight")


# %%
