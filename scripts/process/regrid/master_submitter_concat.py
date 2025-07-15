import os 

for run in ["jed0011", "jed0022", "jed0033"]:
    os.system(f"sbatch scripts/process/regrid/submit_concat.sh {run}")