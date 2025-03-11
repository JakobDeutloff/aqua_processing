#!/bin/bash
#SBATCH --job-name=t_interp # Specify job name
#SBATCH --output=t_interp.o%j # name for standard output log file
#SBATCH --error=t_interp.e%j # name for standard error output log
#SBATCH --partition=compute
#SBATCH --account=bm1183
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --mem=0
set -o errexit -o nounset -o pipefail -o xtrace

# Set pythonpath
export PYTHONPATH="/home/m/m301049/aqua_processing/"

# execute python script in respective environment 
/home/m/m301049/.conda/envs/main/bin/python /home/m/m301049/aqua_processing/scripts/process/interpolate_temp.py $1