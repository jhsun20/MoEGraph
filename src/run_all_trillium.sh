#!/bin/bash

# List of config files to run (edit this list)
CONFIGS=(
  "config/config_HIV_scaffold.yaml --rho_edge 0.45"
  "config/config_HIV_scaffold.yaml --rho_edge 0.55"
  "config/config_HIV_scaffold.yaml --rho_edge 0.65"
  "config/config_HIV_scaffold.yaml --rho_edge 0.75"
)

# Run each config sequentially
for CONFIG_PATH in "${CONFIGS[@]}"; do
    NAME=$(basename "$CONFIG_PATH" .yaml)

    echo "=================================="
    echo "Running experiment: $NAME"
    echo "=================================="

    python src/main.py --config "$CONFIG_PATH"

    # Clear datasets after each run
    # rm -rf /workspace/MoEGraph/datasets/*

    echo ""
done
