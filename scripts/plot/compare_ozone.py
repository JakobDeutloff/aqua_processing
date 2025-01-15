# %%
import xarray as xr
import matplotlib.pyplot as plt
from src.grid_helpers import merge_grid

# %% load data
o3_fixed = xr.open_dataset(
    "/pool/data/ICON/grids/public/mpim/0015/ozone/r0002/bc_ozone_ape.nc", chunks={}
)
v_grid = xr.open_dataset(
    "/work/bm1183/m301049/icon-mpim/experiments/jed0001/jed0001_atm_vgrid_ml.nc"
)
o3_cariolle = xr.open_dataset(
    "/work/bm1183/m301049/icon-mpim/experiments/jed0011/jed0011_atm_3d_main_daymean_19790730T000000Z.14380341.nc",
    chunks={},
).pipe(merge_grid)

# %% calculate mean o3 profile in tropics
mean_o3_fix = (
    o3_fixed.O3.where((o3_fixed.clat < 30) & (o3_fixed.clat > -30))
    .mean(["cell"])
    .compute()
)
mean_o3_car = (
    o3_cariolle.o3.where((o3_cariolle.clat < 30) & (o3_cariolle.clat > -30))
    .mean(["ncells"])
    .compute()
)
mean_p = (
    o3_cariolle.pfull.where((o3_cariolle.clat < 30) & (o3_cariolle.clat > -30))
    .mean(["ncells"])
    .compute()
)
mol_weight_o3 = 48  # g/mol
mol_weight_dry_air = 28.97  # g/mol
mean_o3_fix = mean_o3_fix * (mol_weight_o3 / mol_weight_dry_air)  # convert to kg/kg


# %% plot daily mean o3 cariolle along fixed on pressure levels
fig, ax = plt.subplots(1, 1, figsize=(6, 4), sharey=True)

ax.plot(mean_o3_car.values.squeeze(), mean_p.values.squeeze(), label="cariolle")
ax.plot(mean_o3_fix.squeeze(), mean_o3_fix["plev"], label="fixed", linestyle="--")
ax.set_xlabel("Ozone / kg/kg")
ax.set_ylabel("Pressure / hPa")
ax.legend()
ax.set_yscale("log")
ax.spines[['top', 'right']].set_visible(False)
ax.set_title(o3_cariolle["time"].values[0].astype('datetime64[D]'))
fig.tight_layout()
#fig.savefig(f"plots/ozone_profile_comparison_{o3_cariolle['time'].values[0].astype('datetime64[D]')}.png")

# %%
