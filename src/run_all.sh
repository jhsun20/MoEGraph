#!/bin/bash

# List of config files to run (edit this list)
CONFIGS=(
  "config/config_sst2.yaml"
  "config/config_hiv_scaffold.yaml"
  "config/config_hiv_size.yaml"
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
