#!/usr/bin/env bash
# CPU playback of the 43100 snapshot.
# Fixed 160-grid TRAIN score [34, 123, 39, 136].
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
policy="$root/policy/snapshot_iteration43100.pt"
sha="57f026fdfffeaa7579d9fed8a3ebdd16df3b9f99215038dd02d248c972c09f68"

if [[ ! -f "$policy" ]] || [[ "$(stat -c%s "$policy")" -lt 100000000 ]]; then
  git -C "$root" lfs pull --include="policy/snapshot_iteration43100.pt"
fi
if [[ ! -f "$policy" ]]; then
  echo "Missing policy: $policy" >&2
  exit 1
fi

python="${SCALER_MIMIC_PYTHON:-}"
if [[ -z "$python" ]]; then
  if [[ ! -x "$root/.venv/bin/python" ]]; then
    if command -v python3.10 >/dev/null 2>&1; then
      py=python3.10
    else
      py=python3
    fi
    "$py" -m venv "$root/.venv"
    "$root/.venv/bin/python" -m pip install --upgrade pip
    "$root/.venv/bin/python" -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
    "$root/.venv/bin/python" -m pip install -r "$root/requirements.txt"
  fi
  python="$root/.venv/bin/python"
fi
if [[ ! -x "$python" ]]; then
  echo "Python runtime missing: $python" >&2
  exit 1
fi

unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD CUDA_HOME CUDA_PATH
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$root"
cd -- "$root"
exec "$python" -m scaler_supermimic.viewer \
  --pack "$root/data" \
  --checkpoint "$policy" \
  --checkpoint-sha256 "$sha" \
  --expected-iteration 43100 \
  --visual-urdf "$root/assets/urdf/scaler.urdf" \
  "$@"
