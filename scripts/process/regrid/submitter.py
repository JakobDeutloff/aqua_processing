# %%
import os 

filenames = [
    # "jed1111_atm_2d_19790730T000040Z.17662028.nc",
     "jed1111_atm_3d_ice_19790730T000040Z.17662028.nc",
     "jed1111_atm_3d_liq_19790730T000040Z.17662028.nc",
     "jed1111_atm_3d_main_19790730T000040Z.17662028.nc"
]

for file in filenames:
    os.system(f"sbatch scripts/process/regrid/regrid_coarse.sh {file}")
    os.system(f"sbatch scripts/process/regrid/regrid_fine.sh {file}")


# %%
