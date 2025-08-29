#!/usr/bin/env bash
set -euo pipefail

# Each line: <config_path.yaml> <extra args...>
readarray -t RUNS <<'EOF'
config/config_hiv_size.yaml --rho_edge 0.45
config/config_hiv_size.yaml --rho_edge 0.55
config/config_hiv_size.yaml --rho_edge 0.65
config/config_hiv_size.yaml --rho_edge 0.75
EOF

for LINE in "${RUNS[@]}"; do
  # Tokenize the line into $1..$N
  # shellcheck disable=SC2086
  set -- $LINE

  CFG="$1"; shift             # first token is the config path
  EXTRA_ARGS=("$@")           # the rest are extra args

  # A readable experiment name
  NAME="$(basename "$CFG" .yaml)"
  if ((${#EXTRA_ARGS[@]})); then
    NAME+="__${EXTRA_ARGS[*]// /_}"
  fi

  echo "=================================="
  echo "Running experiment: $NAME"
  echo "=================================="

  # Run: --config gets only the YAML path; extras are passed as normal args
  python src/main.py --config "$CFG" "${EXTRA_ARGS[@]}"

  # Clear datasets after each run
  # rm -rf /workspace/MoEGraph/datasets/*

  echo ""
done
