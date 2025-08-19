#!/bin/bash
#SBATCH --job-name=moegraph
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1        
#SBATCH --time=12:00:00
#SBATCH --account=def-cglee

module purge
module load rdkit/2024.03.5
source moegraphenv/bin/activate


set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# List of config files to run (edit this list)
CONFIGS=(
  "config/config_hiv_size.yaml"
  "config/config_hiv_scaffold.yaml"
  "config/config_motif_size.yaml"
  "config/config_cmnist.yaml"
)

# Run each config sequentially
for CONFIG_PATH in "${CONFIGS[@]}"; do
    NAME=$(basename "$CONFIG_PATH" .yaml)

    echo "=================================="
    echo "Running experiment: $NAME"
    echo "=================================="

    python src/main.py --config "$CONFIG_PATH"

    # Clear datasets after each run
    rm -rf /workspace/MoEGraph/datasets/*

    echo ""
done
