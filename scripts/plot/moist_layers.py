# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# %% load data
runs = ["jed0011", "jed0033", "jed0022"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
colors = {"jed0011": "k", "jed0022": "r", "jed0033": "orange"}
datasets = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_20.nc"
    ).sel(index=slice(0, 2e6))

vgrid = (
    xr.open_dataset(
        "/work/bm1183/m301049/icon-mpim/experiments/jed0001/atm_vgrid_angel.nc"
    )
    .mean("ncells")
    .rename({"height_2": "height", "height": "height_2"})
)


# %% calculate variance of v, q and covariance
def calculate_variances(run, mask, v, q, suffix=None):
    """Helper function to calculate variances and covariance."""
    mean_v = v.where(mask).mean("index")
    mean_q = q.where(mask).mean("index")
    key_suffix = f"_{suffix}" if suffix else ""
    v_var[f"{run}{key_suffix}"] = np.abs(v.where(mask) - mean_v).mean("index")
    q_var[f"{run}{key_suffix}"] = np.abs(q.where(mask) - mean_q).mean("index")
    cov[f"{run}{key_suffix}"] = (
        (v.where(mask) - mean_v) * (q.where(mask) - mean_q)
    ).mean("index")


# Initialize dictionaries
v_var = {}
q_var = {}
cov = {}

# Process each run
for run in runs:
    # Dataset variables
    v = datasets[run]["va"]
    q = datasets[run]["hus"]

    # North
    mask_north = (datasets[run]["clat"] > 0) & (datasets[run]["clat"] < 20)
    calculate_variances(run, mask_north, v, q, suffix="north")

    # South
    mask_south = (datasets[run]["clat"] < 0) & (datasets[run]["clat"] > -20)
    v_adjusted = xr.where(datasets[run]["clat"] < 0, -1 * v, v)
    calculate_variances(run, mask_south, v_adjusted, q, suffix="south")

    # North and South
    mask_north_south = (datasets[run]["clat"] > -20) & (datasets[run]["clat"] < 20)
    calculate_variances(run, mask_north_south, v_adjusted, q)

# %% plot variances
fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharey=True, sharex="col")
for run in runs:
    axes[0, 0].plot(
        v_var[f"{run}_north"],
        vgrid["zg"] / 1e3,
        color=colors[run],
    )
    axes[0, 1].plot(
        q_var[f"{run}_north"],
        vgrid["zg"] / 1e3,
        color=colors[run],
    )
    axes[0, 2].plot(
        cov[f"{run}_north"] * 1e3,
        vgrid["zg"] / 1e3,
        color=colors[run],
    )
    axes[1, 0].plot(
        v_var[f"{run}_south"],
        vgrid["zg"] / 1e3,
        color=colors[run],
    )
    axes[1, 1].plot(
        q_var[f"{run}_south"],
        vgrid["zg"] / 1e3,
        color=colors[run],
    )
    axes[1, 2].plot(
        cov[f"{run}_south"] * 1e3,
        vgrid["zg"] / 1e3,
        color=colors[run],
    )
    axes[2, 0].plot(
        v_var[run],
        vgrid["zg"] / 1e3,
        color=colors[run],
    )
    axes[2, 1].plot(
        q_var[run],
        vgrid["zg"] / 1e3,
        color=colors[run],
    )
    axes[2, 2].plot(
        cov[run] * 1e3,
        vgrid["zg"] / 1e3,
        color=colors[run],
    )

for ax in axes.flatten():
    ax.set_ylim(0, 18)
    ax.spines[["top", "right"]].set_visible(False)

for ax in axes[:, 0]:
    ax.set_ylabel("Height / km")

axes[-1, 0].set_xlabel(r"$\overline{|v'|}$ / m s$^{-1}$")
axes[-1, 0].set_xlim([1, 6])
axes[-1, 1].set_xlabel(r"$\overline{|q'|}$ / kg kg$^{-1}$")
axes[-1, 1].set_xlim([0, 0.0025])
axes[-1, 2].set_xlabel(r"$\overline{v' \cdot q'}$ / 10$^{-3}$ m s$^{-1}$")
axes[0, 0].text(
    -0.2,
    0.5,
    "0 - 20°N",
    ha="center",
    va="center",
    rotation=90,
    color="k",
    transform=axes[0, 0].transAxes,
)
axes[1, 0].text(
    -0.2,
    0.5,
    "0 - 20S",
    ha="center",
    va="center",
    rotation=90,
    color="k",
    transform=axes[1, 0].transAxes,
)
axes[2, 0].text(
    -0.2,
    0.5,
    "20°S - 20°N",
    ha="center",
    va="center",
    rotation=90,
    color="k",
    transform=axes[2, 0].transAxes,
)


fig.legend(
    handles=axes[0, 0].lines,
    labels=[exp_name[run] for run in runs],
    bbox_to_anchor=(0.63, 0),
    ncols=3,
)
fig.tight_layout()
fig.savefig("plots/misc/qv_variance_20_20.png", dpi=300, bbox_inches="tight")

# %%
