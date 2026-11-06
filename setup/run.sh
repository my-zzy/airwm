#!/usr/bin/env bash
# Use the pinned Conda environment without an inherited system CUDA library path.
set -euo pipefail
if [[ "$#" -eq 0 ]]; then
    echo "Usage: bash setup/run.sh airwm <doctor|collect|profile|train|evaluate> ..." >&2
    exit 2
fi
exec env -u LD_LIBRARY_PATH conda run --no-capture-output --name airwm "$@"
