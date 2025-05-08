#!/bin/bash
set -o errexit -o nounset -o pipefail -o xtrace

# Set pythonpath
export PYTHONPATH="/home/m/m301049/aqua_processing"

runs=("jed0011" "jed0022" "jed0033")

for run in "${runs[@]}"; do
    echo "Processing run: ${run}"
    #/home/m/m301049/.conda/envs/main/bin/python /home/m/m301049/aqua_processing/scripts/conceptual_model/calculate_albedo.py ${run}
    #/home/m/m301049/.conda/envs/main/bin/python /home/m/m301049/aqua_processing/scripts/conceptual_model/calculate_emissivity.py ${run}
    #/home/m/m301049/.conda/envs/main/bin/python /home/m/m301049/aqua_processing/scripts/conceptual_model/calculate_lt_param.py ${run}
    /home/m/m301049/.conda/envs/main/bin/python /home/m/m301049/aqua_processing/scripts/conceptual_model/run.py ${run}
done