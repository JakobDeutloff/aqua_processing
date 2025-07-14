# %%
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from src.read_data import load_definitions, load_vgrid

# %% load data
ds = xr.open_dataset(
    f"/work/bm1183/m301049/icon_hcap_data/control/production/random_sample/jed0011_randsample_processed_64.nc"
).sel(index=slice(0, 1e6))
cre = xr.open_dataset(
    f"/work/bm1183/m301049/icon_hcap_data/control/production/cre/jed0011_cre_raw.nc"
)
vgrid = load_vgrid()
runs, exp_name, colors, line_labels, sw_color, lw_color, net_color, linestyles = (
    load_definitions()
)
# %% calculate cf
iwp_bins = np.logspace(-4, 2, 51)
iwp_points = (iwp_bins[:-1] + iwp_bins[1:]) / 2

cf_ice = (ds["cli"] + ds["qs"] + ds["qg"]) > 5e-7
cf_liq = xr.where(ds["conn"] == 1, (ds["qr"] + ds["clw"]), 0) > 5e-7
cf_ice_binned = cf_ice.groupby_bins(ds["iwp"], iwp_bins).mean()
cf_liq_binned = cf_liq.groupby_bins(ds["iwp"], iwp_bins).mean()

# %% plot cfs and cre
fig, axes = plt.subplots(
    4,
    1,
    figsize=(8, 10),
    sharex="col",
    height_ratios=[3, 2, 1, 1],
)

# cf
# Create a colormap from blue to yellow
blue_to_yellow = LinearSegmentedColormap.from_list("BlueToYellow", ["yellow", "blue"])

# Create a colormap from red to yellow
red_to_yellow = LinearSegmentedColormap.from_list("RedToYellow", ["yellow", "red"])

cont2 = axes[0].contourf(
    iwp_points,
    vgrid["zg"] / 1e3,
    cf_liq_binned.values.T,
    levels=np.linspace(0.1, 1, 10),
    cmap="autumn_r",
    alpha=1,
)
cont = axes[0].contourf(
    iwp_points,
    vgrid["zg"] / 1e3,
    cf_ice_binned.values.T,
    levels=np.linspace(0.1, 1, 10),
    cmap="winter_r",
    alpha=0.7,
)
axes[0].set_ylim([0, 17])
axes[0].set_yticks([0, 6, 15])
axes[0].set_ylabel("Height / km")

# cre
axes[1].axhline(0, color="k", lw=0.5, ls="-")
axes[1].plot(cre["iwp"], cre["sw"], color=sw_color, label="SW")
axes[1].plot(cre["iwp"], cre["lw"], color=lw_color, label="LW")
axes[1].plot(cre["iwp"], cre["net"], color=net_color, label="Net")
axes[1].set_ylabel(r"$C(I)$ / W m$^{-2}$")
axes[1].legend(
    loc="lower left",
    frameon=False,
)
axes[1].set_yticks([-250, 0, 200])

# P(I)
hist, edges = np.histogram(
    ds["iwp"].values,
    bins=np.logspace(-4, np.log10(40), 51),
    density=False,
)
hist = hist / len(ds["iwp"].values)
axes[2].stairs(
    hist,
    edges,
    color="k",
)
axes[2].set_yticks([0, 0.02])
axes[2].set_ylabel(r"$P(I)$")

# C * P
axes[3].fill_between(
    edges[1:],
    0,
    hist * cre["net"].values,
    step="pre",
    color="gray",
)
axes[3].axhline(0, color="k", lw=0.5, ls="-")
axes[3].stairs(
    hist * cre["net"].values,
    edges,
    color="k",
)
axes[3].set_yticks([-0.5, 0, 0.5])
axes[3].set_ylabel(r"$C(I) \cdot P(I)$ / W m$^{-2}$")
axes[3].set_xlabel(r"$I$ / kg m$^{-2}$")

# Get the position of the main axis
box = axes[0].get_position()

# Width and height of each colorbar
cb_width = 0.02
cb_height = 0.45 * (box.y1 - box.y0)  # 40% of axis height
cb_pad = 0.02

# Colorbar 1 (top)
cb_ax1 = fig.add_axes(
    [box.x1 + cb_pad, box.y0 + cb_height + cb_pad, cb_width, cb_height]
)
cb_ice = fig.colorbar(cont, cax=cb_ax1)
cb_ice.set_ticks([0.1, 0.5, 1])
cb_ice.set_label("Ice Cloud Fraction")

# Colorbar 2 (bottom)
cb_ax2 = fig.add_axes([box.x1 + cb_pad, box.y0, cb_width, cb_height])
cb_liq = fig.colorbar(cont2, cax=cb_ax2)
cb_liq.set_ticks([0.1, 0.5, 1])
cb_liq.set_label("Liquid Cloud Fraction")


axes[0].set_xscale("log")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim([1e-4, 1e1])

fig.savefig("plots/publication/overview.png", dpi=300, bbox_inches="tight")
# %%
