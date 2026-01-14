# %%
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from src.read_data import (
    load_cre,
    load_definitions,
    load_iwp_hists,
    load_random_datasets,
)
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

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
        axes[0, i].set_title(f"{names[name]}")
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

# %% multiply CRE and iwp hist
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

# %% calculate integrated CRE and feedback
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

fig.savefig("plots/publication/sup_feedback_sensitivity.pdf", bbox_inches="tight")

# %%
lc_fraction = {}
mask_fraction = {}
for run in runs:
    lc_fraction[run] = (
        (datasets[run]["lwp"] > 1e-4)
        .groupby_bins(datasets[run]["iwp"], bins=np.logspace(-4, 1, 50))
        .mean()
    )
    mask_fraction[run] = (
        datasets[run]["mask_low_cloud"]
        .groupby_bins(datasets[run]["iwp"], bins=np.logspace(-4, 1, 50))
        .mean()
    )

# %% plot low cloud fractions
fig, ax = plt.subplots()
for run in runs:
    ax.plot(
        np.logspace(-4, 1, 49),
        lc_fraction[run],
        label=definitions[3][run],
        color=definitions[2][run],
    )
    ax.plot(
        np.logspace(-4, 1, 49),
        mask_fraction[run],
        linestyle="--",
        color=definitions[2][run],
    )
ax.set_xscale("log")


# %% plot iwp resolved feedback
fig, ax = plt.subplots()
for case in cases:
    ax.plot(
        const_cre_folded[case]["jed0022"]["iwp"],
        const_cre_folded[case]["jed0022"]["net"]
        - const_cre_folded[case]["jed0011"]["net"],
        label=case,
    )
ax.set_xscale("log")
ax.legend()
# %% plot CRE changes in one plot
fig = plt.figure(figsize=(10, 8))
gs = GridSpec(2, 2, figure=fig)
linestyles = {
    "raw": "-",
    "wetsky": "--",
    "clearsky": ":",
}

# Create the main 1x2 axes
ax00 = fig.add_subplot(gs[0, 0])

# Now split axes[0, 1] into two (vertical split)
gs_sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1])
ax10 = fig.add_subplot(gs_sub[0, 0])
ax20 = fig.add_subplot(gs_sub[1, 0])

ax00.axhline(0, color="k", linewidth=0.5)
ax10.axhline(0, color="k", linewidth=0.5)
ax20.axhline(0, color="k", linewidth=0.5)
for case in cases:
    for part in parts:
        ax00.plot(
            cre[case]["jed0011"]["iwp"],
            cre[case]["jed0011"][part],
            color=colors[part],
            linestyle=linestyles[case],
        )
    ax10.plot(
        cre[case]["jed0033"]["iwp"],
        (cre[case]["jed0033"]["lw"] - cre[case]["jed0011"]["lw"]),
        color=colors["lw"],
        linestyle=linestyles[case],
    )
    ax20.plot(
        cre[case]["jed0033"]["iwp"],
        (cre[case]["jed0033"]["sw"] - cre[case]["jed0011"]["sw"]),
        color=colors["sw"],
        linestyle=linestyles[case],
    )
for ax in [ax00, ax10, ax20]:
    ax.set_xscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(1e-4, 10)

for ax in [ax10, ax20]:
    ax.set_ylim([-2, 10])
ax10.set_xticklabels("")
ax00.set_ylabel(r"$C(I)$ / W m$^{-2}$")
ax10.set_ylabel(r"$\Delta C_{\mathrm{LW}}(I)$ / W m$^{-2}$")
ax20.set_ylabel(r"$\Delta C_{\mathrm{SW}}(I)$ / W m$^{-2}$")
ax20.set_xlabel("$I$ / kg m$^{-2}$")
ax00.set_xlabel("$I$ / kg m$^{-2}$")

handles = [
    plt.Line2D([0], [0], color=colors["lw"], label="LW"),
    plt.Line2D([0], [0], color=colors["sw"], label="SW"),
    plt.Line2D([0], [0], color=colors["net"], label="Net"),
    plt.Line2D([0], [0], color="grey", linestyle="-", label="High Clouds"),
    plt.Line2D([0], [0], color="grey", linestyle="--", label="Frozen Clouds"),
    plt.Line2D([0], [0], color="grey", linestyle=":", label="All Clouds"),
]
fig.legend(
    handles=handles,
    bbox_to_anchor=(0.65, 0.45),
    ncol=2,
    frameon=False,
)
fig.savefig("plots/publication/sup_cre_sensitivity.pdf", bbox_inches="tight")

# %% make feedback plot with feedback as function of iwp
fig, axes = plt.subplots(
    2,
    2,
    figsize=(8, 6),
    sharex="col",
    sharey="col",
    gridspec_kw={"width_ratios": [3, 1]},
)

for ax in axes.flatten():
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", linewidth=0.5)

for case in cases:
    axes[0, 0].plot(
        const_iwp_folded[case]["jed0033"]["iwp"],
        (
            const_iwp_folded[case]["jed0033"]["lw"]
            - const_iwp_folded[case]["jed0011"]["lw"]
        )
        / 2,
        color=colors["lw"],
        linestyle=linestyles[case],
    )
    axes[0, 0].plot(
        const_iwp_folded[case]["jed0033"]["iwp"],
        (
            const_iwp_folded[case]["jed0033"]["sw"]
            - const_iwp_folded[case]["jed0011"]["sw"]
        )
        / 2,
        color=colors["sw"],
        linestyle=linestyles[case],
    )
    axes[1, 0].plot(
        const_cre_folded[case]["jed0033"]["iwp"],
        (
            const_cre_folded[case]["jed0033"]["net"]
            - const_cre_folded[case]["jed0011"]["net"]
        )
        / 2,
        color=colors["net"],
        linestyle=linestyles[case],
    )
    axes[0, 1].scatter(
        0,
        feedback_const_iwp[case]["jed0033"]["lw"],
        color=colors["lw"],
        marker=markers[case],
    )
    axes[0, 1].scatter(
        0.5,
        feedback_const_iwp[case]["jed0033"]["sw"],
        color=colors["sw"],
        marker=markers[case],
    )
    axes[1, 1].scatter(
        1,
        feedback_const_cre[case]["jed0033"]["net"],
        color=colors["net"],
        marker=markers[case],
    )

for ax in axes[:, 0]:
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 10)

axes[1, 0].set_xlabel("$I$ / kg m$^{-2}$")
axes[0, 0].set_ylabel(r"$F_{\mathrm{C}}(I)$ / W m$^{-2}$ K$^{-1}$")
axes[1, 0].set_ylabel(r"$F_{\mathrm{P}}(I)$ / W m$^{-2}$ K$^{-1}$")
axes[0, 1].set_ylabel(r"$F_{\mathrm{C}}$ / W m$^{-2}$ K$^{-1}$")
axes[1, 1].set_ylabel(r"$F_{\mathrm{P}}$ / W m$^{-2}$ K$^{-1}$")

for ax in axes[:, 1]:
    ax.set_yticks([-0.05, 0, 0.15, 0.25])
    ax.set_xticks([0, 0.5, 1], ["LW", "SW", "Net"])

fig.tight_layout()
handles = [
    plt.Line2D([0], [0], color=colors["lw"], label="LW"),
    plt.Line2D([0], [0], color=colors["sw"], label="SW"),
    plt.Line2D([0], [0], color=colors["net"], label="Net"),
    plt.Line2D([0], [0], color="grey", linestyle="-", label="High Clouds"),
    plt.Line2D([0], [0], color="grey", linestyle="--", label="Frozen Clouds"),
    plt.Line2D([0], [0], color="grey", linestyle=":", label="All Clouds"),
]
fig.legend(
    handles=handles,
    bbox_to_anchor=(0.6, 0),
    ncol=2,
    frameon=False,
)
handles = [
    plt.Line2D(
        [0], [0], color="grey", linestyle="", marker=markers["raw"], label="High Clouds"
    ),
    plt.Line2D(
        [0],
        [0],
        color="grey",
        linestyle="",
        marker=markers["wetsky"],
        label="Frozen Clouds",
    ),
    plt.Line2D(
        [0],
        [0],
        color="grey",
        linestyle="",
        marker=markers["clearsky"],
        label="All Clouds",
    ),
]
fig.legend(
    handles,
    ["High Clouds", "Frozen Clouds", "All Clouds"],
    ncol=1,
    bbox_to_anchor=(0.95, 0),
    frameon=False,
)
fig.savefig("plots/publication/sup_feedback_sensitivity.pdf", bbox_inches="tight")

# %% try finding the cause for decrease of SW CRE at low I
iwp_range = slice(1e-4, 1e-1)
mean_cs = {}
mean_ws = {}
mean_as = {}
for run in runs:
    mean_cs[run] = (
        datasets[run]["rsutcs"]
        .groupby_bins(datasets[run]["iwp"], bins=np.logspace(-4, 1, 50))
        .mean()
    )
    mean_ws[run] = (
        datasets[run]["rsutws"]
        .groupby_bins(datasets[run]["iwp"], bins=np.logspace(-4, 1, 50))
        .mean()
    )
    mean_as[run] = (
        datasets[run]["rsut"]
        .groupby_bins(datasets[run]["iwp"], bins=np.logspace(-4, 1, 50))
        .mean()
    )

# %% plot mean fluxes
fig, axes = plt.subplots(3, 2, figsize=(6, 8), sharex=True, sharey="col")
for run in runs:
    axes[0, 0].plot(
        np.logspace(-4, 1, 49),
        mean_cs[run],
        label=definitions[3][run],
        color=definitions[2][run],
    )
    axes[1, 0].plot(
        np.logspace(-4, 1, 49),
        mean_ws[run],
        label=definitions[3][run],
        color=definitions[2][run],
    )
    axes[2, 0].plot(
        np.logspace(-4, 1, 49),
        mean_as[run],
        label=definitions[3][run],
        color=definitions[2][run],
    )
for run in ["jed0033"]:
    axes[0, 1].plot(
        np.logspace(-4, 1, 49),
        (mean_cs[run] - mean_cs["jed0011"]) * -1,
        label=definitions[3][run],
        color=definitions[2][run],
    )
    axes[1, 1].plot(
        np.logspace(-4, 1, 49),
        (mean_ws[run] - mean_ws["jed0011"]) * -1,
        label=definitions[3][run],
        color=definitions[2][run],
    )
    axes[2, 1].plot(
        np.logspace(-4, 1, 49),
        (mean_as[run] - mean_as["jed0011"]) * -1,
        label=definitions[3][run],
        color=definitions[2][run],
    )
for ax in axes.flatten():
    ax.set_xscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(1e-4, 1e0)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8)

axes[0, 0].set_ylim(30, 150)
axes[0, 1].set_ylim(-2, 2)

# %%
fig, ax = plt.subplots()
ax.plot(
    np.logspace(-4, 1, 49),
    (mean_as["jed0022"] - mean_as["jed0011"]) * -1
    - (mean_ws["jed0022"] - mean_ws["jed0011"]) * -1,
    label="Frozen Clouds",
    color=definitions[2]["jed0022"],
)
ax.set_xscale("log")
ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
ax.set_ylim(-2, 2)

# %% plot diurnal cycle between 1e-2 and 1e-1
data = {}
for run in runs:
    data[run] = datasets[run]["time_local"].where(
        (datasets[run]["iwp"] >= 1e-2) & (datasets[run]["iwp"] < 1e-1)
    )

hists = {}
for run in runs:
    hists[run], bin_edges = np.histogram(
        data[run], bins=np.arange(0, 25, 1), density=False
    )

# %% calculate mean incoming SW radiation
for run in runs:
    mean_rsdt = (
        datasets[run]["rsdt"]
        .where((datasets[run]["iwp"] >= 1e-2) & (datasets[run]["iwp"] < 1e-1))
        .mean()
        .values
    )
    print(f"Mean incoming SW radiation for {definitions[3][run]}: {mean_rsdt:.2f} W/m2")


# %%
fig, ax = plt.subplots()
for run in runs:
    ax.plot(
        np.arange(0, 24, 1),
        hists[run],
        label=definitions[3][run],
        color=definitions[2][run],
    )

# %%
