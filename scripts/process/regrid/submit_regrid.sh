#!/bin/bash
#SBATCH --job-name=concat_2d
#SBATCH --output=concat%j.out
#SBATCH --error=concat%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --partition=compute
#SBATCH --account=bm1183

set -o errexit -o nounset -o pipefail -o xtrace

# Set pythonpath
export PYTHONPATH="/home/m/m301049/aqua_processing/"

# execute python script in respective environment 
/home/m/m301049/.conda/envs/main/bin/python /home/m/m301049/aqua_processing/scripts/process/regrid/calc_zonal_sections.py