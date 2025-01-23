#!/bin/bash
#SBATCH --job-name=weights # Specify job name
#SBATCH --output=weights.o%j # name for standard output log file
#SBATCH --error=weights.e%j # name for standard error output log
#SBATCH --partition=compute
#SBATCH --account=bm1183
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --mem=0
set -o errexit -o nounset -o pipefail -o xtrace

# Set pythonpath
export PYTHONPATH="/home/m/m301049/aqua_processing/"

# execute python script in respective environment 
/home/m/m301049/.conda/envs/main/bin/python /home/m/m301049/aqua_processing/scripts/process/calc_tropical_mean.py