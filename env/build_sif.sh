#!/usr/bin/env bash
# Run from repository root on the Vega login/build host with internet.
set -euo pipefail
SIF=${1:-env/neuromri.sif}
RUNTIME=${RUNTIME:-apptainer}
mkdir -p "$(dirname "$SIF")"
"$RUNTIME" build --fakeroot "$SIF" env/neuromri.def
sif_dir=$(cd "$(dirname "$SIF")" && pwd)
sif_name=$(basename "$SIF")
(cd "$sif_dir" && sha256sum "$sif_name" | tee "$sif_name.sha256")
"$RUNTIME" exec "$SIF" python -m pip freeze > "${SIF%.sif}.pip-freeze.txt"
echo "Archive SIF, checksum and freeze externally; keep the SIF for all reported runs."
