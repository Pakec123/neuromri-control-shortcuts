#!/usr/bin/env bash
# Sourced by slurm/runtime.sh inside an allocation when DATA_TAR is set.
# Vega compute nodes see high per-file latency on the shared shared storage home (an epoch over
# ~7k small images did not finish in 7 min), so the frozen dataset tar is verified
# against its SHA256 and unpacked once per job and node onto node-local NVMe.
# Concurrent job steps on the same node share one copy through flock.
local_root="${NEUROMRI_LOCAL_ROOT:-/tmp/${USER}/nmri-${SLURM_JOB_ID:?}}"
staged="$local_root/data"
mkdir -p "$local_root"
exec 9>"$local_root/.lock"
flock 9
if [[ ! -f "$staged/.staged" ]]; then
    expected=$(awk '{print $1}' "$DATA_TAR.sha256")
    actual=$(sha256sum "$DATA_TAR" | awk '{print $1}')
    [[ "$expected" == "$actual" ]] || { echo "Dataset tar SHA256 mismatch: $DATA_TAR" >&2; exit 2; }
    rm -rf "$staged.tmp"
    mkdir -p "$staged.tmp"
    tar -xf "$DATA_TAR" -C "$staged.tmp"
    mv "$staged.tmp" "$staged"
    echo "$actual" > "$staged/.staged"
fi
flock -u 9
exec 9>&-
DATA_ROOT="$staged/${DATA_TAR_SUBDIR:-Brain Disease}"
export NEUROMRI_DATA_TAR_SHA256
NEUROMRI_DATA_TAR_SHA256=$(cat "$staged/.staged")
