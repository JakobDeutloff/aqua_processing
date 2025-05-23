"""
Model run of the conceptual model used in the study
"""
#%%
import numpy as np
from src.hc_model import run_model
import xarray as xr
import pickle
from src.read_data import load_parameters, load_lt_quantities
import sys
import os 

#%% Set CWD to PYTHONPATH
pythonpath = os.environ.get("PYTHONPATH", "")
if pythonpath:
    os.chdir(pythonpath)
    print(f"Working directory set to: {os.getcwd()}")
else:
    print("PYTHONPATH is not set.")

# %% load data
run = sys.argv[1]
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
ds = xr.open_dataset(
    f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed.nc"
).sel(index=slice(None, 1e6))
parameters = load_parameters(run)
lt_quantities = load_lt_quantities(run)
with open(f"data/{run}_lw_vars_mean.pkl", "rb") as f:
    hc_em = pickle.load(f)

# %% set mask ans bins 
mask = True
IWP_bins = np.logspace(-4, 1, num=50)

# %% calculate constants used in the model
SW_in = 424 # ds["rsdt"].where(mask).mean().values
parameters["SW_in"] = SW_in


# %% run model 
result = run_model(
    IWP_bins,
    T_hc = ds["hc_top_temperature"].where(mask),
    LWP = ds['lwp'].where(mask),
    IWP = ds['iwp'].where(mask),
    connectedness=ds['conn'],
    parameters = parameters,
    prescribed_lc_quantities=lt_quantities,
    #prescribed_hc_em=hc_em,
)
# %% save result 
with open(f'data/model_output/{run}.pkl', 'wb') as f:
    pickle.dump(result, f)
# %%
