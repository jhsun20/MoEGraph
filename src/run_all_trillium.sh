#!/bin/bash
#SBATCH --job-name=moegraph
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --account=def-cglee
#SBATCH -o logs/%x-%j.out
#SBATCH --chdir=/scratch/jhsun/MoEGraph   # ensures we run from your project dir

set -euo pipefail

module purge
module load StdEnv/2023 gcc/12.3
module load rdkit/2024.03.5               # only if you actually need RDKit

# activate env in this directory
source moegraphenv/bin/activate

# keep CPU threads in check
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python - <<'PY'
import torch
torch.set_num_threads(1); torch.set_num_interop_threads(1)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))
PY

# Configs to run
CONFIGS=(
  config/config_hiv_size.yaml
  config/config_hiv_scaffold.yaml
  config/config_motif_size.yaml
  config/config_cmnist.yaml
)

for CONFIG_PATH in "${CONFIGS[@]}"; do
  NAME=$(basename "$CONFIG_PATH" .yaml)
  echo "=================================="
  echo "Running experiment: $NAME"
  echo "=================================="

  # use srun so SLURM tracks resources properly
  srun python -u src/main.py --config "$CONFIG_PATH"

  # Clear datasets after each run (adjust if your code writes elsewhere)
  if [ -d "datasets" ]; then
    rm -rf datasets/*
  fi

  echo
done
