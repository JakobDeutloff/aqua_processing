# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
config = ["const_o3", "cariolle", "cariolle", "cariolle"]  # configuration
n_nodes = [128, 128, 256, 512]  # number of nodes
integrate_nh = [2604.578, 2824.513, 1501.684, 878.158]  # computation length in s
write_output = [198.046, 302.598, 460.663, 284.770]  # write output length in s
sim_length = [1, 1, 1, 1]  # simulation length in days


# %%
def calc_nh_per_day(n_nodes, integrate_nh, sim_length):
    return (integrate_nh / 60 / 60) * n_nodes / sim_length


def calc_nodes_for_month(n_nodes, integrate_nh, sim_length):
    return calc_nh_per_day(n_nodes, integrate_nh, sim_length) * 31 / 8


def calc_simulated_days_8h(integrate_hn, sim_length):
    return (sim_length / (integrate_hn / 60 / 60)) * 8


# %% plot
colors = ["grey", "k", "k", "k"]
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
for i, (conf, n, nh, oh, sl, col) in enumerate(
    zip(config, n_nodes, integrate_nh, write_output, sim_length, colors)
):
    axes[0].bar(i, calc_nh_per_day(n, nh + oh, sl), label=conf, color=col)
    axes[0].set_ylabel("Node hours per day")
    axes[1].bar(i, calc_nodes_for_month(n, nh + oh, sl), label=conf, color=col)
    axes[1].set_ylabel("Nodes for one month")
    axes[2].bar(i, calc_simulated_days_8h(nh + oh, sl), label=conf, color=col)
    axes[2].set_ylabel("Simulated days / 8h")

for ax in axes:
    ax.set_xticks(range(len(n_nodes)))
    ax.set_xticklabels(n_nodes)
    ax.set_xlabel("Number of nodes")
    ax.spines[["top", "right"]].set_visible(False)
fig.legend(
    axes[0].get_legend_handles_labels()[0][:2],
    config[:2],
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.1),
)
fig.savefig("plots/nodes_hours_with_output.png", bbox_inches="tight")


# %% make log performance plot
fig, ax = plt.subplots()
for i, (conf, n, nh, sl) in enumerate(
    zip(config[1:], n_nodes[1:], integrate_nh[1:], sim_length[1:])
):
    ax.scatter(n, calc_simulated_days_8h(nh, sl), label=conf, color="k")

ax.plot(
    n_nodes[1:],
    (calc_simulated_days_8h(integrate_nh[1], sim_length[1]) * np.array(n_nodes[1:]))
    / n_nodes[1],
    label="O(n)",
    color="grey",
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(n_nodes[1:])
ax.set_xticklabels(n_nodes[1:])
ax.set_xlabel("Number of nodes")
ax.set_ylabel("Simulated days / 8h")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig("plots/nodes_hours_with_output.png", bbox_inches="tight")


# %%
