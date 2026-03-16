# %%
import matplotlib.pyplot as plt
import numpy as np
import pickle
from src.read_data import (
    load_cre,
    load_definitions,
    load_iwp_hists,
    load_random_datasets,
)

# %% load data
cre = {
    "raw": load_cre(name="raw"),
    "wetsky": load_cre(name="wetsky"),
    "clearsky": load_cre(name="clearsky"),
}
cre_conn = {
    "5": load_cre(name="raw"),
    "1": load_cre(name="conn_1"),
    "20": load_cre(name="conn_20"),
}
definitions = load_definitions()
histograms = load_iwp_hists()
datasets = load_random_datasets(version="processed")

# %% make comparison plot of CREs for control run
fig, ax = plt.subplots()
run = "jed0011"
color_sw = definitions[4]
color_lw = definitions[5]
color_net = definitions[6]
linestyles = {
    "5": "-",
    "1": "--",
    "20": ":",
}
names = {
    "5": "5 %",
    "1": "1 %",
    "20": "20 %",
}

for name, cre_data in cre_conn.items():
    ax.plot(
        cre_data[run]["iwp"],
        cre_data[run]["sw"],
        label=f"SW {names[name]}",
        color=color_sw,
        linestyle=linestyles[name],
    )
    ax.plot(
        cre_data[run]["iwp"],
        cre_data[run]["lw"],
        label=f"LW {names[name]}",
        color=color_lw,
        linestyle=linestyles[name],
    )
    ax.plot(
        cre_data[run]["iwp"],
        cre_data[run]["net"],
        label=f"Net {names[name]}",
        color=color_net,
        linestyle=linestyles[name],
    )

ax.set_xscale("log")
ax.set_xlim(1e-4, 10)
ax.spines[["top", "right"]].set_visible(False)


# %% plot difference of CREs for all cases
fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey="row")
linestyles = {
    "jed0022": "-",
    "jed0033": "--",
}

for i, name in enumerate(["raw", "wetsky", "clearsky"]):
    for run in ["jed0022", "jed0033"]:
        axes[0, i].set_title(f"{name}")
        axes[0, i].axhline(0, color="k", linestyle="-", linewidth=0.8)
        axes[1, i].axhline(0, color="k", linestyle="-", linewidth=0.8)
        axes[0, i].plot(
            cre[name][run]["iwp"],
            (cre[name][run]["lw"] - cre[name]["jed0011"]["lw"]),
            color=color_lw,
            linestyle=linestyles[run],
        )
        axes[1, i].plot(
            cre[name][run]["iwp"],
            (cre[name][run]["sw"] - cre[name]["jed0011"]["sw"]),
            color=color_sw,
            linestyle=linestyles[run],
        )

for ax in axes.flatten():
    ax.set_xscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(1e-4, 10)
    ax.set_ylim(-3, 27)


# %% calculate feedback values
runs = ["jed0011", "jed0022", "jed0033"]
cases = ["5", "1", "20"]

cre_folded = {}
const_iwp_folded = {}
const_cre_folded = {}

for case in cases:
    cre_folded[case] = {}
    const_iwp_folded[case] = {}
    const_cre_folded[case] = {}
    for run in runs:
        cre_folded[case][run] = cre_conn[case][run] * histograms[run]
        const_iwp_folded[case][run] = cre_conn[case][run] * histograms["jed0011"]
        const_cre_folded[case][run] = cre_conn[case]["jed0011"] * histograms[run]

#  calculate integrated CRE and feedback
cre_integrated = {}
cre_const_iwp_integrated = {}
cre_const_cre_integrated = {}
feedback = {}
feedback_const_iwp = {}
feedback_const_cre = {}

for case in cases:
    cre_integrated[case] = {}
    cre_const_iwp_integrated[case] = {}
    cre_const_cre_integrated[case] = {}
    feedback[case] = {}
    feedback_const_iwp[case] = {}
    feedback_const_cre[case] = {}

    for run in runs:
        cre_integrated[case][run] = cre_folded[case][run].sum(dim="iwp")
        cre_const_iwp_integrated[case][run] = const_iwp_folded[case][run].sum(dim="iwp")
        cre_const_cre_integrated[case][run] = const_cre_folded[case][run].sum(dim="iwp")

    feedback[case]["jed0033"] = (
        cre_integrated[case]["jed0033"] - cre_integrated[case]["jed0011"]
    ) / 2
    feedback[case]["jed0022"] = (
        cre_integrated[case]["jed0022"] - cre_integrated[case]["jed0011"]
    ) / 4
    feedback_const_iwp[case]["jed0033"] = (
        cre_const_iwp_integrated[case]["jed0033"]
        - cre_const_iwp_integrated[case]["jed0011"]
    ) / 2
    feedback_const_iwp[case]["jed0022"] = (
        cre_const_iwp_integrated[case]["jed0022"]
        - cre_const_iwp_integrated[case]["jed0011"]
    ) / 4
    feedback_const_cre[case]["jed0033"] = (
        cre_const_cre_integrated[case]["jed0033"]
        - cre_const_cre_integrated[case]["jed0011"]
    ) / 2
    feedback_const_cre[case]["jed0022"] = (
        cre_const_cre_integrated[case]["jed0022"]
        - cre_const_cre_integrated[case]["jed0011"]
    ) / 4


# %% plot feedback values
parts = ["lw", "sw", "net"]
feedbacks = {
    "cre": feedback_const_iwp,
    "iwp": feedback_const_cre,
    "Total Feedback": feedback,
}
offsets = {
    "lw": 0,
    "sw": 1,
    "net": 2,
}
colors = {
    "lw": definitions[5],
    "sw": definitions[4],
    "net": definitions[6],
}
markers = {
    "5": "o",
    "1": "s",
    "20": "^",
}
fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharey=True, sharex=True)
for i, f_name in enumerate(feedbacks.keys()):
    for case in cases:
        for part in parts:
            axes[i].scatter(
                offsets[part],
                feedbacks[f_name][case]["jed0022"][part],
                color=colors[part],
                marker=markers[case],
                facecolor="none",
            )

for ax in axes:
    ax.set_xticks([0, 1, 2], ["LW", "SW", "Net"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8)

axes[0].set_ylabel(r"$F_{\mathrm{C}}$ / W m$^{-2}$ K$^{-1}$")
axes[1].set_ylabel(r"$F_{\mathrm{P}}$ / W m$^{-2}$ K$^{-1}$")
axes[2].set_ylabel(r"$F$ / W m$^{-2}$ K$^{-1}$")
handles = [
    plt.Line2D([0], [0], color="grey", linestyle="", marker=markers["1"], label="1 %"),
    plt.Line2D([0], [0], color="grey", linestyle="", marker=markers["5"], label="5 %"),
    plt.Line2D(
        [0], [0], color="grey", linestyle="", marker=markers["20"], label="20 %"
    ),
]
fig.legend(
    handles,
    [handle.get_label() for handle in handles],
    bbox_to_anchor=(0.7, 0),
    ncol=3,
    frameon=False,
)

# label all plots 
for j, ax in enumerate(axes):
    ax.text(
        0.1,
        1.05,
        chr(97 + j),
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="right",
    )

fig.savefig("plots/publication/feedback_sensitivity.pdf", bbox_inches="tight")

# %% calculate low cloud fractions
lwp_fraction = {}
lc_fraction = {}
cong_fraction = {}
iwp_bins = np.logspace(-4, np.log10(40), 51)
iwp_points = (iwp_bins[:-1] + iwp_bins[1:]) / 2
for run in runs:
    lwp_fraction[run] = (
        (datasets[run]["lwp"] > 1e-4)
        .groupby_bins(datasets[run]["iwp"], bins=iwp_bins)
        .mean()
    )
    lc_fraction[run] = (
        datasets[run]["mask_low_cloud"]
        .groupby_bins(datasets[run]["iwp"], bins=iwp_bins)
        .mean()
    )
    cong_fraction[run] = (
        ((datasets[run]["lwp"] > 1e-4) & (datasets[run]["conn"] == 1))
        .groupby_bins(datasets[run]["iwp"], bins=iwp_bins)
        .mean()
    )

# %% plot connected LWP fraction
fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
for run in runs:
    axes[0].plot(
        np.logspace(-4, 1, 49),
        lwp_fraction[run],
        color=definitions[2][run],
        label=definitions[3][run],
    )

    axes[1].plot(
        np.logspace(-4, 1, 49),
        lc_fraction[run],
        color=definitions[2][run],
        label=definitions[3][run],
    )

    axes[2].plot(
        np.logspace(-4, 1, 49),
        cong_fraction[run],
        color=definitions[2][run],
        label=definitions[3][run],
    )
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1e-1)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel('Liquid CF')
axes[0].set_ylim(0.42, 0.57)
axes[1].set_ylabel('Low CF')
axes[1].set_ylim(0.42, 0.57)
axes[2].set_ylabel('Congestus CF')
axes[2].set_ylim(0, 0.15)
axes[2].set_xlabel('$I$ / kg m$^{-2}$')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    bbox_to_anchor=(0.7, 0),
    ncol=3,
    frameon=False,
)
# label all axes
for j, ax in enumerate(axes):
    ax.text(
        0.05,
        1.05,
        chr(97 + j),
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="right",
    )

fig.savefig("plots/publication/lcf_comparison.pdf", bbox_inches="tight")

# %% mean lwp of congestus 
mean_congestus_lwp = {}
for run in runs:
    mean_congestus_lwp[run] = (
        datasets[run]["lwp"]
        .where(datasets[run]["conn"] == 1)
        .median()
    )

    print(f"Mean LWP of congestus for {definitions[3][run]}: {mean_congestus_lwp[run].values:.4f} kg/m2")

#%% plot plot SW cre diff for wetsky
fig, ax = plt.subplots(figsize=(7, 3))
ax.axhline(0, color="k", linewidth=0.5)
ax.plot(
    cre["wetsky"]["jed0022"]["iwp"],
    cre["wetsky"]["jed0022"]["sw"] - cre["wetsky"]["jed0011"]["sw"],
    color=colors["sw"],
    linestyle=linestyles["jed0022"],
    label='+4 K'
)
ax.plot(
    cre["wetsky"]["jed0033"]["iwp"],
    cre["wetsky"]["jed0033"]["sw"] - cre["wetsky"]["jed0011"]["sw"],
    color=colors["sw"],
    linestyle=linestyles["jed0033"],
    label='+2 K'
)

ax.spines[["top", "right"]].set_visible(False)
ax.set_xscale("log")
ax.set_xlim(1e-4, 10)
ax.set_ylim(-3, 27)
ax.set_yticks([0, 5, 25])
ax.set_xlabel("$I$ / kg m$^{-2}$")
ax.set_ylabel(r"$\Delta C_{\mathrm{SW}}(I)$ / W m$^{-2}$")
ax.legend(frameon=False)
fig.savefig("plots/publication/sw_cre_comparison.pdf", bbox_inches="tight")


# %% save data 
path = '/work/bm1183/m301049/icon_hcap_data/publication/lc_fraction'
for run in runs:
    cong_fraction[run]['iwp_bins'] = iwp_points
    lwp_fraction[run]['iwp_bins'] = iwp_points
    cong_fraction[run].to_netcdf(f"{path}/{run}_cong_fraction.nc")
    lwp_fraction[run].to_netcdf(f"{path}/{run}_lwp_fraction.nc")

for run in runs:
    cre['wetsky'][run].to_netcdf(f"/work/bm1183/m301049/icon_hcap_data/publication/cre/{run}_cre_wetsky.nc")
    cre_conn['1'][run].to_netcdf(f"/work/bm1183/m301049/icon_hcap_data/publication/cre/{run}_cre_conn_1.nc")
    cre_conn['20'][run].to_netcdf(f"/work/bm1183/m301049/icon_hcap_data/publication/cre/{run}_cre_conn_20.nc")


# %%
