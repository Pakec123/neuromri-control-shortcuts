#!/usr/bin/env bash
# External-only settings; internal P1/P2 runtime is unchanged.
set -euo pipefail
source slurm/runtime.sh
: "${EXTERNAL_RUN_DIR:?Use tools/run_external.sh}"
source "$EXTERNAL_RUN_DIR/run.env"
CONTAINER+=(--bind "$EXTERNAL_RAW_ROOT:$EXTERNAL_RAW_ROOT:ro" --bind "$EXTERNAL_ROOT:$EXTERNAL_ROOT:ro"
    --bind "$EXTERNAL_TOOLS:$EXTERNAL_TOOLS:ro")
export APPTAINERENV_HF_HUB_OFFLINE=1 APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
# This digest belongs to the preprocessing SIF only in preprocessing commands.
PREPROCESS_SHA=$(sha256sum "$EXTERNAL_SIF" | awk '{print $1}')
"${CONTAINER[@]}" "$SIF" python tools/external_run_manifest.py verify --run "$EXTERNAL_RUN_DIR"
