#!/usr/bin/env bash
# Run after: conda create -n airwm python=3.12 pip -y; conda activate airwm
set -euo pipefail
airwm_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CONDA_PREFIX:?Activate the airwm Conda environment first}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "airwm" ]]; then
    echo "Activate the dedicated airwm environment before installing." >&2
    exit 1
fi
python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
if [[ "${AIRWM_DIRECT:-0}" == "1" ]]; then
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi
airwm_indexes=(--extra-index-url https://pypi.nvidia.com)
if [[ -n "${AIRWM_PYPI_MIRROR:-}" ]]; then
    airwm_indexes+=(--extra-index-url "$AIRWM_PYPI_MIRROR")
fi
airwm_indexes+=(--default-index https://pypi.org/simple)
python -m pip install uv==0.13.7
UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}" UV_HTTP_RETRIES=5 UV_CONCURRENT_DOWNLOADS=4 \
python -m uv pip install --python "$CONDA_PREFIX/bin/python" \
    -r "$airwm_root/setup/environment/airwm-python-linux-64.lock" --require-hashes \
    --overrides "$airwm_root/setup/environment/isaaclab-resolved-overrides.txt" \
    "${airwm_indexes[@]}" --torch-backend cu130 \
    --index-strategy unsafe-best-match
python -m uv pip install --python "$CONDA_PREFIX/bin/python" \
    --no-deps --no-build-isolation -e "$airwm_root"

