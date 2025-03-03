# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# %% load data
runs = ["jed0011", "jed0022", "jed0033"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
datasets = {}
for run in runs:
    datasets[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample.nc"
    )

vgrid = xr.open_dataset(
    "/work/bm1183/m301049/icon-mpim/experiments/jed0001/atm_vgrid_angel.nc"
).mean("ncells")
# %% define constants
cp = 1005  # J kg^-1 K^-1
g = 9.81  # m s^-2

# %% calculate heating rates
sw_hr = (
    (1 / (datasets["jed0011"]["rho"] * cp))
    * (datasets["jed0011"]["rsd"] - datasets["jed0011"]["rsu"]).diff("height")
    / (vgrid["zg"].diff('height_2').values)
)

# %%
lw_hr = (1 / (datasets["jed0011"]["rho"] * cp)) * (
    (datasets["jed0011"]["rld"] - datasets["jed0011"]["rlu"]).diff('height')
    / (vgrid["zg"].diff('height_2').values)
)

# %%
iwp_bins = np.logspace(-4, np.log10(40), 51)
iwp_points = (iwp_bins[1:] + iwp_bins[:-1]) / 2
sw_hr_bin = sw_hr.groupby_bins(
    datasets["jed0011"]["clivi"] + datasets["jed0011"]["qsvi"], bins=iwp_bins
).mean()
lw_hr_bin = lw_hr.groupby_bins(
    datasets["jed0011"]["clivi"] + datasets["jed0011"]["qsvi"], bins=iwp_bins
).mean()

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharex=True, sharey=True)
col = axes[0].pcolormesh(iwp_points, vgrid["zghalf"][1:-1][40:], sw_hr_bin.T[40:, :])
col2 = axes[1].pcolormesh(iwp_points, vgrid["zghalf"][1:-1][40:], lw_hr_bin.T[40:, :])
fig.colorbar(col, ax=axes[0])
fig.colorbar(col2, ax=axes[1])
axes[0].set_xscale("log")

# %%
fig, axes = plt.subplots(2, 4, figsize=(12, 8), sharey=True)

axes[0, 0].plot(datasets["jed0011"]["rsu"].mean("index"), vgrid["zg"])
axes[0, 0].set_title("rsu")
axes[0, 1].plot(datasets["jed0011"]["rsd"].mean("index"), vgrid["zg"])
axes[0, 1].set_title("rsd")
axes[0, 2].plot(datasets["jed0011"]["rlu"].mean("index"), vgrid["zg"])
axes[0, 2].set_title("rlu")
axes[0, 3].plot(datasets["jed0011"]["rld"].mean("index"), vgrid["zg"])
axes[0, 3].set_title("rld")

axes[1, 0].plot(sw_hr.mean("index"), vgrid["zghalf"][1:-1])
axes[1, 0].set_title("sw_hr")
axes[1, 1].plot(lw_hr.mean("index"), vgrid["zghalf"][1:-1])
axes[1, 1].set_title("lw_hr")



# %%
