# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import datetime as dt
from concurrent.futures import ProcessPoolExecutor
from src.moist_layers import plot_prw_field, get_contour_length
from functools import partial

# mpl.use("WebAgg")  # Use WebAgg backend for interactive plotting

# %% load data
ds = xr.open_dataset(
    "/work/bm1183/m301049/icon_hcap_data/control/production/latlon/prw_latlon.nc"
)
ds = ds.sel(time=(ds.time.dt.minute == 0) & (ds.time.dt.hour == 0)).sel(
    lon=slice(0, 100), lat=slice(-50, 50)
)

# %% check histogram

hists = []
intervals = [5, 10, 15, 20, 25, 30, 35, 40]
for interval in intervals:
    hist, edges = np.histogram(
        ds["prw"].isel(time=slice(0, 20)).sel(lat=slice(-interval, interval)),
        density=False,
        bins=np.arange(0, 60, 1),
    )
    hists.append(hist)

# %%
fig, ax = plt.subplots(figsize=(8, 5))
for hist, interval in zip(hists, intervals):
    ax.plot(edges[:-1], hist, label=f"-{interval}° to {interval}°")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(title="Latitude band")
ax.set_xlabel("Water vapor path / mm")
ax.set_ylabel("Counts")
fig.savefig("plots/moist_layers/prw_histograms_sec.png", dpi=300, bbox_inches="tight")

# %% calculate contourlength 30 mm
def contour_length_for_time(t):
    # Helper for parallel execution
    return int(get_contour_length(ds["prw"].sel(time=t), cont=30))


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
    attrs={"long_name": "Contour length of 30 mm PRW"},
)


# %% calculate contourlength 24 mm
def contour_length_for_time(t):
    # Helper for parallel execution
    return int(get_contour_length(ds["prw"].sel(time=t), cont=20))


with ProcessPoolExecutor(max_workers=64) as executor:
    contour_lenghts = list(
        tqdm(
            executor.map(contour_length_for_time, ds["time"].values),
            total=len(ds["time"]),
        )
    )

contour_lenghts_24 = xr.DataArray(
    contour_lenghts / np.max(contour_lenghts),
    dims=["time"],
    coords={"time": ds["prw"].time},
    attrs={"long_name": "Contour length of 24 mm PRW"},
)

# %% plot contour length 30mm
fig, ax = plt.subplots(figsize=(10, 5))
contour_lenghts_30.plot(ax=ax, color="k", label="Contour length")
ax.set_xlabel("Time")
ax.set_ylabel("Normalized 30 mm contour length")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.axvline(x=dt.datetime.strptime("1979-07-02", "%Y-%m-%d"), color="grey")
ax.axvline(x=dt.datetime.strptime("1979-07-17", "%Y-%m-%d"), color="grey")
fig.savefig(
    "plots/moist_layers/contour_length_30_sec.png", dpi=300, bbox_inches="tight"
)

# %% plot contour length 24mm
fig, ax = plt.subplots(figsize=(10, 5))
contour_lenghts_24.plot(ax=ax, color="k", label="Contour length")
ax.set_xlabel("Time")
ax.set_ylabel("Normalized 30 mm contour length")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.axvline(x=dt.datetime.strptime("1979-07-14", "%Y-%m-%d"), color="grey")
ax.axvline(x=dt.datetime.strptime("1979-09-23", "%Y-%m-%d"), color="grey")
fig.savefig(
    "plots/moist_layers/contour_length_24_sec.png", dpi=300, bbox_inches="tight"
)

# %% correlate contour lenths
corr = xr.corr(contour_lenghts_24, contour_lenghts_30)
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(
    contour_lenghts_24,
    contour_lenghts_30,
    color="k",
    s=5,
)
ax.set_title(f"Correlation: {corr.values:.2f}")
ax.set_xlabel("Normalized 24 mm contour length")
ax.set_ylabel("Normalized 30 mm contour length")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig(
    "plots/moist_layers/contour_length_correlation_sec.png",
    dpi=300,
    bbox_inches="tight",
)

# %%
fig, ax = plot_prw_field(ds["prw"].sel(time="1979-07-02"), 30)
fig.savefig("plots/moist_layers/prw_field_1979-07-02.png", dpi=300, bbox_inches="tight")
fig, ax = plot_prw_field(ds["prw"].sel(time="1979-07-17"), 30)
fig.savefig("plots/moist_layers/prw_field_1979-07-17.png", dpi=300, bbox_inches="tight")


# %% get contour lengths for differnet thresholds 
thresholds = np.arange(20, 41, 1)

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


# %% correlate clengths 
corrs = np.zeros((len(thresholds), len(thresholds)))
for i, cont1 in enumerate(thresholds):
    for j, cont2 in enumerate(thresholds):
        corr = xr.corr(c_lentghts[str(cont1)], c_lentghts[str(cont2)])
        corrs[i, j] = corr.values

# %% plot correlation matrix
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.pcolormesh(corrs, cmap="viridis", vmin=0, vmax=1)
fig.colorbar(cax, ax=ax, label="Correlation coefficient")

ax.set_xticks(np.arange(len(thresholds)) + 0.5, labels=thresholds)
ax.set_yticks(np.arange(len(thresholds)) + 0.5, labels=thresholds)
ax.set_xlabel("Contour / mm")
ax.set_ylabel("Contour / mm")
fig.savefig(
    "plots/moist_layers/contour_length_correlation_matrix_sec.png",
    dpi=300,
    bbox_inches="tight",
)

# %%
