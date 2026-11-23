#!/usr/bin/env bash
set -euo pipefail
source slurm/external_runtime.sh
: "${EXTERNAL_ROOT:?}"
CONTAINER+=(--bind "$EXTERNAL_ROOT:$EXTERNAL_ROOT:ro")
name=${1:?encoder}
bash slurm/gpu_step.sh "${CONTAINER[@]}" --nv "$SIF" python -m neuromri.extract_embeddings \
    --set "encoder=$name" "output=${EXTERNAL_SOURCE_CACHE:-cache/external_source_vega}"
bash slurm/gpu_step.sh "${CONTAINER[@]}" --nv "$SIF" python -m neuromri.external_embeddings \
    --set "encoder=$name" "external_root=$EXTERNAL_ROOT" \
    "external_manifest=$EXTERNAL_MANIFEST_DIR/slices.csv" \
    "external_embeddings=${EXTERNAL_CACHE:-cache/external_vega}"
