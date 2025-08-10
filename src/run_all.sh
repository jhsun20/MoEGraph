#!/bin/bash

# List of config files to run (edit this list)
CONFIGS=(
  "config/config_div_sparse_cmnist.yaml"
  "config/config_div_sparse_motif.yaml"
)

# Run each config sequentially
for CONFIG_PATH in "${CONFIGS[@]}"; do
    NAME=$(basename "$CONFIG_PATH" .yaml)

    echo "=================================="
    echo "Running experiment: $NAME"
    echo "=================================="

    python src/main.py --config "$CONFIG_PATH"

    echo ""
done
