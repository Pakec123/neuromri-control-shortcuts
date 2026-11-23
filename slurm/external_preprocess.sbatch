#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source slurm/external_runtime.sh
# Writable processed root; raw stays read-only. No --nv, no network downloads.
PRE=(apptainer exec --cleanenv --bind "$REPO:$REPO" --bind "$EXTERNAL_RAW_ROOT:$EXTERNAL_RAW_ROOT:ro"
     --bind "$EXTERNAL_ROOT:$EXTERNAL_ROOT" --bind "$EXTERNAL_TOOLS:$EXTERNAL_TOOLS:ro" --pwd "$REPO")
"${PRE[@]}" "$EXTERNAL_SIF" python -m neuromri.external_data inventory \
    --root "$EXTERNAL_RAW_ROOT" --manifest-dir "$EXTERNAL_MANIFEST_DIR" \
    --extracted-root "$EXTERNAL_RUN_DIR/volumes"
APPTAINERENV_NEUROMRI_CONTAINER_SHA256="$PREPROCESS_SHA" "${PRE[@]}" "$EXTERNAL_SIF" \
    python -m neuromri.external_data preprocess --cpu --scope vega \
    --root "$EXTERNAL_RUN_DIR/volumes" --manifest-dir "$EXTERNAL_MANIFEST_DIR" \
    --output-root "$EXTERNAL_ROOT" --tools "$EXTERNAL_TOOLS"
"${CONTAINER[@]}" "$SIF" python -m neuromri.verify_data
"${CONTAINER[@]}" "$SIF" python tools/figure_external.py --source-root "$DATA_ROOT" \
    --external-root "$EXTERNAL_ROOT" --manifest "$EXTERNAL_MANIFEST_DIR/slices.csv" \
    --output "$EXTERNAL_RUN_DIR/qc/input_montage.png"
"${PRE[@]}" "$EXTERNAL_SIF" python tools/audit_external.py --root "$EXTERNAL_RAW_ROOT" \
    --processed-root "$EXTERNAL_ROOT" --manifest-dir "$EXTERNAL_MANIFEST_DIR" \
    --output "$EXTERNAL_RUN_DIR/qc"
