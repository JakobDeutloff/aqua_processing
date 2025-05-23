# %%
import matplotlib.pyplot as plt
import pickle
import xarray as xr

# %% load data
runs = ["jed0011", "jed0033", "jed0022"]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
colors = {"jed0011": "k", "jed0022": "red", "jed0033": "orange"}
linestyles = {
    "jed0011": "-",
    "jed0022": "--",
    "jed0033": ":",
}
results = {}
cres = {}
for run in runs:
    with open(f"data/model_output/{run}.pkl", "rb") as f:
        results[run] = pickle.load(f)
    cres[run] = xr.open_dataset(
            f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/cre/{run}_cre_raw.nc"
        )

# %% make plot wit all quantities
fig, axes = plt.subplots(5, 2, figsize=(10, 10), sharex=True)

for run in runs:
    # temperature
    axes[0, 0].plot(results[run].index, results[run]['T_hc'], color=colors[run], label=exp_name[run], alpha=0.5)
    # emissivity 
    axes[0, 1].plot(results[run].index, results[run]['em_hc'], color=colors[run], label=exp_name[run])
    # lc_frac 
    axes[1, 0].plot(results[run].index, results[run]['lc_fraction'], color=colors[run], label=exp_name[run], alpha=0.5)
    # albedo
    axes[1, 1].plot(results[run].index, results[run]['alpha_hc'], color=colors[run], label=exp_name[run])
    # lt upwelling LW 
    axes[2, 0].plot(results[run].index, results[run]['R_t'], color=colors[run], label=exp_name[run])
    # lt albedo
    axes[2, 1].plot(results[run].index, results[run]['alpha_t'], color=colors[run], label=exp_name[run])
    # cre 
    axes[3, 0].plot(cres[run]['iwp'], cres[run]['sw'], color='blue', label=exp_name[run], linestyle=linestyles[run])
    axes[3, 0].plot(cres[run]['iwp'], cres[run]['lw'], color='red', label=exp_name[run], linestyle=linestyles[run])
    axes[3, 0].plot(cres[run]['iwp'], cres[run]['net'], color='black', label=exp_name[run], linestyle=linestyles[run])
    # cre predicted
    axes[3, 1].plot(results[run].index, results[run]['SW_cre'], color='blue', label=exp_name[run], linestyle=linestyles[run])
    axes[3, 1].plot(results[run].index, results[run]['LW_cre'], color='red', label=exp_name[run], linestyle=linestyles[run])
    axes[3, 1].plot(results[run].index, results[run]['SW_cre'] + results[run]['LW_cre'], color='black', label=exp_name[run], linestyle=linestyles[run])

for run in ['jed0022', 'jed0033']:
    # CRE diff
    axes[4, 0].plot(cres[run]['iwp'], cres[run]['sw'] - cres['jed0011']['sw'], color='blue', label=exp_name[run], linestyle=linestyles[run])
    axes[4, 0].plot(cres[run]['iwp'], cres[run]['lw'] - cres['jed0011']['lw'], color='red', label=exp_name[run], linestyle=linestyles[run])
    # CRE predicted diff
    axes[4, 1].plot(results[run].index, results[run]['SW_cre'] - results['jed0011']['SW_cre'], color='blue', label=exp_name[run], linestyle=linestyles[run])
    axes[4, 1].plot(results[run].index, results[run]['LW_cre'] - results['jed0011']['LW_cre'], color='red', label=exp_name[run], linestyle=linestyles[run])

axes[0, 0].set_ylabel("HC Temperature / K")
axes[0, 1].set_ylabel("HC Emissivity")
axes[1, 0].set_ylabel("Low Cloud Fraction")
axes[1, 1].set_ylabel("HC Albedo")
axes[2, 0].set_ylabel("Upwelling LW / W m$^{-2}$")
axes[2, 1].set_ylabel("LT Albedo")
axes[3, 0].set_ylabel("CRE / W m$^{-2}$")
axes[3, 1].set_ylabel("CRE CM / W m$^{-2}$")
axes[4, 0].set_ylabel("CRE - CRE Control / W m$^{-2}$")
axes[4, 1].set_ylabel("CRE CM - CRE CM Control / W m$^{-2}$")

axes[4, 1].sharey(axes[4, 0])
axes[4, 0].set_ylim(-3, 7)
axes[4, 0].set_xlabel("IWP / kg m$^{-2}$")
axes[4, 1].set_xlabel("IWP / kg m$^{-2}$")



axes[0, 0].set_xscale("log")
axes[0, 0].set_xlim([1e-4, 1e1])
for ax in axes.flatten():
    ax.spines[['top', 'right']].set_visible(False)

fig.tight_layout()

# %% direct comparison CRE
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
for i, run in enumerate(runs):

    axes[i].plot(cres[run]['iwp'], cres[run]['sw'], color='blue')
    axes[i].plot(cres[run]['iwp'], cres[run]['lw'], color='red')
    axes[i].plot(cres[run]['iwp'], cres[run]['net'], color='black')
    axes[i].plot(results[run].index, results[run]['SW_cre'], color='blue', linestyle='--')
    axes[i].plot(results[run].index, results[run]['LW_cre'], color='red', linestyle='--')
    axes[i].plot(results[run].index, results[run]['SW_cre'] + results[run]['LW_cre'], color='black', linestyle='--')
    axes[i].set_title(exp_name[run])
    axes[i].set_xlabel("IWP / kg m$^{-2}$")
    axes[i].set_ylabel("CRE / W m$^{-2}$")
    axes[i].set_xscale("log")
    axes[i].set_xlim([1e-4, 1e1])
    axes[i].spines[['top', 'right']].set_visible(False)

fig.tight_layout()


# %% compare lw cre and cre from fat 
import numpy as np
fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharex=True)

for run in runs[1:]:
    ax.plot(cres[run]['iwp'], cres[run]['lw'] - cres['jed0011']['lw'], color=colors[run], label=exp_name[run], linestyle='-')

    ax.plot(results[run].index, np.abs(results[run]['R_t'] - results['jed0011']['R_t']) - (5.67e-8 * (results[run]['T_hc']**4 - results['jed0011']['T_hc']**4)), color=colors[run], linestyle='--')


ax.set_xscale("log")
ax.set_xlim(1e-1, 1e1)
# %%
