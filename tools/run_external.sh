#!/usr/bin/env bash
# ONE login-node command: stage all network inputs, then CPU -> GPU -> CPU.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:---upload}
[[ $# -le 1 && ( "$MODE" == --download || "$MODE" == --upload ) ]] || {
    echo 'Usage: bash tools/run_external.sh [--download|--upload]' >&2; exit 2;
}
[[ -z "$(git status --porcelain)" ]] || { echo 'Commit code/manifests before Vega staging'; exit 2; }
source "${NEUROMRI_HPC_ENV:-slurm/hpc.env}"
[[ ! -f slurm/hpc.local.env ]] || source slurm/hpc.local.env
for key in ACCOUNT PARTITION CPU_PARTITION DATA_ROOT REPO SIF WEIGHTS_ROOT; do
    value=${!key:-}
    [[ -n "$value" && "$value" != *'<TBD>'* ]] || { echo "Set $key in slurm/hpc.local.env" >&2; exit 2; }
done
[[ -f "$SIF" ]] || { echo "Wait for the main SIF build: $SIF"; exit 2; }
[[ -f "$SIF.sha256" ]] || sha256sum "$SIF" > "$SIF.sha256"
actual=$(sha256sum "$SIF" | awk '{print $1}')
[[ "$actual" == "$(awk '{print $1}' "$SIF.sha256")" ]] || { echo 'Main SIF checksum mismatch'; exit 2; }
EXTERNAL_RAW_ROOT=${EXTERNAL_RAW_ROOT:-${EXTERNAL_RAW_ROOT:?Set EXTERNAL_RAW_ROOT}}
EXTERNAL_TOOLS=${EXTERNAL_TOOLS:-${EXTERNAL_TOOLS:?Set EXTERNAL_TOOLS}}
EXTERNAL_SIF=${EXTERNAL_SIF:-$REPO/env/neuromri-external.sif}
EXTERNAL_RUN_DIR=${EXTERNAL_RUN_DIR:-$REPO/results/p2_external_vega/$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD)}
[[ "$EXTERNAL_RUN_DIR" = /* ]] || { echo 'EXTERNAL_RUN_DIR must be absolute'; exit 2; }
[[ ! -e "$EXTERNAL_RUN_DIR" ]] || { echo 'Choose a fresh EXTERNAL_RUN_DIR'; exit 2; }
EXTERNAL_ROOT="$EXTERNAL_RUN_DIR/processed"
EXTERNAL_MANIFEST_DIR="$EXTERNAL_RUN_DIR/manifests"
EXTERNAL_SOURCE_CACHE="$EXTERNAL_RUN_DIR/cache/source"
EXTERNAL_CACHE="$EXTERNAL_RUN_DIR/cache/external"
EXTERNAL_OUTPUT="$EXTERNAL_RUN_DIR/probes"
mkdir -p "$EXTERNAL_ROOT" "$EXTERNAL_MANIFEST_DIR" "$WEIGHTS_ROOT" "$EXTERNAL_TOOLS" logs
mkdir -p "$EXTERNAL_RAW_ROOT"
# Login-node internet only; raw is writable here, read-only in all later jobs.
if [[ "$MODE" == --download ]]; then
    apptainer exec --cleanenv --bind "$REPO:$REPO" \
        --bind "$EXTERNAL_RAW_ROOT:$EXTERNAL_RAW_ROOT" --pwd "$REPO" "$SIF" \
        python -m neuromri.external_download --root "$EXTERNAL_RAW_ROOT"
fi
# Container build and ALL downloads occur here on the login node.
[[ -f "$EXTERNAL_SIF" ]] || apptainer build --fakeroot --ignore-subuid --ignore-fakeroot-command "$EXTERNAL_SIF" env/neuromri-external.def
[[ -f "$EXTERNAL_SIF.sha256" ]] || sha256sum "$EXTERNAL_SIF" > "$EXTERNAL_SIF.sha256"
[[ "$(sha256sum "$EXTERNAL_SIF" | awk '{print $1}')" == "$(awk '{print $1}' "$EXTERNAL_SIF.sha256")" ]]
LOGIN=(apptainer exec --cleanenv --bind "$REPO:$REPO" --bind "$WEIGHTS_ROOT:$WEIGHTS_ROOT"
       --bind "$EXTERNAL_TOOLS:$EXTERNAL_TOOLS" --bind "$EXTERNAL_RAW_ROOT:$EXTERNAL_RAW_ROOT:ro" --pwd "$REPO")
"${LOGIN[@]}" "$SIF" python -m neuromri.fetch_weights --dest "$WEIGHTS_ROOT" \
    --encoders resnet18 resnet50 dinov2_s biomedclip
"${LOGIN[@]}" "$SIF" python tools/fetch_external_tools.py --output "$EXTERNAL_TOOLS"
for key in EXTERNAL_RAW_ROOT EXTERNAL_ROOT EXTERNAL_MANIFEST_DIR EXTERNAL_TOOLS EXTERNAL_SIF \
           EXTERNAL_SOURCE_CACHE EXTERNAL_CACHE EXTERNAL_OUTPUT; do
    printf '%s=%q\n' "$key" "${!key}"
done > "$EXTERNAL_RUN_DIR/run.env"
git ls-files -z > "$EXTERNAL_RUN_DIR/tracked_files.nul"
"${LOGIN[@]}" "$SIF" python tools/external_run_manifest.py stage --run "$EXTERNAL_RUN_DIR" \
    --git-commit "$(git rev-parse HEAD)" --tracked-list "$EXTERNAL_RUN_DIR/tracked_files.nul" \
    --raw-root "$EXTERNAL_RAW_ROOT" --sif "$SIF" --external-sif "$EXTERNAL_SIF" \
    --weights "$WEIGHTS_ROOT" --tools "$EXTERNAL_TOOLS"
export EXTERNAL_RUN_DIR
bash slurm/submit_external.sh
