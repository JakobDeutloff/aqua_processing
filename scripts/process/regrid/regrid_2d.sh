#!/bin/bash
#SBATCH --job-name=icon_regrid_fine
#SBATCH --output=regrid_fine_%j.out
#SBATCH --error=regrid_fine_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --partition=compute
#SBATCH --account=bm1183

set -o errexit -o nounset -o pipefail -o xtrace

module load cdo

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

INPATH=$1
FILE="prw_icon.nc"
FILE_PATH="$INPATH/$FILE"

# Directories
TARGET_GRID="/work/bm1183/m301049/icon_hcap_data/control/production/latlon/target_grid_fine_global.txt"

# Generate weights only once
WEIGHTS_FILE="/work/bm1183/m301049/icon_hcap_data/control/production/latlon/weights_global.nc"
if [ ! -f "$WEIGHTS_FILE" ]; then
    echo "⚙️  Generating weights"
    cdo -P $OMP_NUM_THREADS gencon,"$TARGET_GRID" "$FILE_PATH" "$WEIGHTS_FILE"
fi

# Regrid entire file in one pass
OUTFILE="$INPATH/prw_latlon.nc"
cdo -P $OMP_NUM_THREADS remap,"$TARGET_GRID","$WEIGHTS_FILE" "$FILE_PATH" "$OUTFILE"

# delete temporary files
rm $FILE_PATH

echo "✅ Done with $FILE → $OUTFILE"

