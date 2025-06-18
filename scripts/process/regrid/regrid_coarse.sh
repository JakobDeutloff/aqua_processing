#!/bin/bash
#SBATCH --job-name=icon_regrid_coarse
#SBATCH --output=regrid_coarse_%j.out
#SBATCH --error=regrid_coarse_%j.err
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

# Directories
INPUT_DIR=/work/bm1183/m301049/icon-mpim/experiments/jed1111/
OUTDIR=/work/bm1183/m301049/icon_hcap_data/control/production/latlon/
FILE=$1
FILE_PATH="$INPUT_DIR/$FILE"
BASENAME="${FILE%.nc}"
TARGET_GRID="$OUTDIR/target_grid_coarse.txt"

echo "🚀 Processing $FILE..."

# Generate weights only once
WEIGHTS_FILE="$OUTDIR/weights_coarse.nc"
if [ ! -f "$WEIGHTS_FILE" ]; then
    echo "⚙️  Generating weights from first timestep of main file..."
    cdo -P $OMP_NUM_THREADS gencon,"$TARGET_GRID" "$FILE_PATH" "$WEIGHTS_FILE"
fi

# Regrid entire file in one pass
OUTFILE="$OUTDIR/${BASENAME}_coarse.nc"
cdo -P $OMP_NUM_THREADS remap,"$TARGET_GRID","$WEIGHTS_FILE" "$FILE_PATH" "$OUTFILE"

echo "✅ Done with $FILE → $OUTFILE"
done
