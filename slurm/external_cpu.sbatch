#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source slurm/external_runtime.sh
# Probes consume authenticated caches/manifests only, no raw scans or GPU.
srun "${CONTAINER[@]}" "$SIF" python -m neuromri.p2_external --set \
    "jobs=$CPU_CPUS" "threads_per_fit=1" \
    "source_embeddings=${EXTERNAL_SOURCE_CACHE:-cache/external_source_vega}" \
    "external_embeddings=${EXTERNAL_CACHE:-cache/external_vega}" \
    "external_manifest=$EXTERNAL_MANIFEST_DIR/slices.csv" \
    "external_subjects=$EXTERNAL_MANIFEST_DIR/subjects.csv" \
    "output=${EXTERNAL_OUTPUT:-results/p2_external_vega}"
srun "${CONTAINER[@]}" "$SIF" python -m neuromri.external_diagnostics --set \
    "source_embeddings=$EXTERNAL_SOURCE_CACHE" "external_embeddings=$EXTERNAL_CACHE" \
    "external_manifest=$EXTERNAL_MANIFEST_DIR/slices.csv" "external_root=$EXTERNAL_ROOT" \
    "output=$EXTERNAL_OUTPUT"
"${CONTAINER[@]}" "$SIF" python tools/external_run_manifest.py complete --run "$EXTERNAL_RUN_DIR"
