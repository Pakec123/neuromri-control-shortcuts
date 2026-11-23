# Reproducing external threshold transport

This public run guide replaces the development-machine operations notes. It
preserves the E1–E4 analysis commands and frozen configuration; no data, results,
credentials or machine-specific deployment instructions are distributed.

1. Obtain NeuroMRI v1 and the external releases listed in the README. Review the
   publisher terms. Keep archives outside git and verify `downloads.csv` hashes.
2. Build `env/neuromri.def` and `env/neuromri-external.def` with Apptainer on a
   permitted build host. Record each SIF SHA-256 and its `/opt/neuromri/pip-freeze.txt`.
3. Copy `slurm/hpc.local.env.example` to `slurm/hpc.local.env`. Supply absolute
   paths, the allocation and GPU/CPU partition names. Commit the public source
   and frozen manifests before staging. Keep the local environment file ignored.
4. On the HPC login host, run `bash tools/run_external.sh --download`. This
   downloads public archives, verifies byte counts/hashes, fetches locked tools
   and encoder weights, creates a fresh result directory and stages its manifest.
   Use `--upload` if you have already transferred the exact verified archives.
5. The launcher submits `external_preprocess.sbatch` → `external_extract.sbatch`
   → `external_cpu.sbatch`, linked by `afterok`. The first inventories and
   preprocesses anatomy on CPU, the second recomputes source and external
   embeddings using four independent A100 steps, and the last fits scorers,
   computes E1–E4 paired statistics and diagnostics, then completes the manifest.
6. Accept output only after `RESULTS_MANIFEST.json` has state `complete` and all
   hashes verify: `python tools/external_run_manifest.py verify --run "$RUN"`.
   Numerical tables are `$RUN/probes/E1_metrics.csv`, `E1_contrasts.csv`,
   `E2_metrics.csv`, `E2_tests.csv`, `E3_metrics.csv`, `E3_contrasts.csv`,
   `E4_metrics.csv` and `E4_tests.csv`. Diagnostics and exported role inventories
   live alongside them. All four encoders and unavailable-budget rows are retained.

`configs/p2_external.yaml` fixes E1 NeuroMRI→PD cohorts, E2 cross-cohort PD,
E3 IXI specificity and E4 external AD. It also fixes score-blind age matching,
AD control roles, calibration draws and the separate 8/24/8 test families.
Rebuilding images can change platform-dependent file hashes: retain newly
produced pipeline provenance and do not overwrite the original frozen manifests.
The historical `scope: vega` value is an analysis identity label; an equivalent
HPC must supply SLURM, the SIF hash and an A100 for GPU work. It is not a hostname.

See `PUBLIC_SNAPSHOT.md` for the limits of historical attestation verification.
No historical provenance is rewritten or promoted by this guide.
