<!-- Public sanitized rendition; site paths and obsolete operational references removed. Analysis specification unchanged. -->
# P2 internal integrity-gate amendment

Dated 20 November 2026. Written AFTER the P2 internal Vega export
(`results/p2/`, SLURM job 50579635) was transferred, but BEFORE any P2 internal
endpoint was computed or inspected in the manuscript workflow. This is a
post-export acceptance rule for commit identity only. It does not change the
P2 results plan (`docs/P2_RESULTS_PLAN.md`, last changed at commit fc07982),
its strategies, budgets, draws, seeds, reference fit or any estimand. It applies
the pattern of the P1 amendment (`docs/P1_PROVENANCE_ATTESTATION.md`) to a
single-job export; it is implemented by `tools/p2_internal_gate.py` using the
shared helpers in `tools/attestation_common.py`.

## Why an amendment is needed

The Vega main SIF and compute nodes have no git binary. `provenance.json`
records the exact error `[Errno 2] No such file or directory: 'git'` in both
`git_commit` and `git_dirty`. The original fields are kept unchanged. The fix in
main commit 808ca5e (`git_commit_files`, `source_fingerprint`) postdates this job
and is not applied retroactively. An earlier attempt (job 50546892) produced no
results and is not part of this export.

## Machine checks (all must pass; any failure blocks P2 internal analysis)

1. **Transfer archive.** `p2_results.tar` has SHA-256
   `4051b422b8f6ad9779fac775f4df130517438c49784a8ead7e07cf8f3d4402b1`; its five
   members are byte-identical to the extracted `metrics.csv`, `predictions.csv`,
   `image_predictions.csv`, `provenance.json` and `pip-freeze.txt`.
2. **Scheduler and reflog.** The attestation text collected on Vega
   (`attestation/p2_internal_attestation.txt`) records HEAD
   ada9d384411f23bddac563532763c8f31a868413 as the newest reflog entry
   (fast-forward at 2026-11-20T08:26:14+01:00) immediately after the P1
   production commit 469ee8f1aa6f5bbacaf6dc3d078b9920a01b5941. Job 50579635 is
   COMPLETED with exit code 0:0 on a named node with the single production
   WorkDir `<ORIGINAL_INTERNAL_WORKDIR>`. Submit, start and
   end (Europe/Ljubljana) lie after the HEAD change; because sacct already reports
   the end time and HEAD was still ada9d38 when the attestation was collected,
   the whole job ran at ada9d38. The provenance SLURM identity equals the job.
3. **Executed source.** The P2-relevant diff (code, configuration, manifests,
   weight registry, SIF definition/lock and SLURM scripts listed in the tool)
   from 469ee8f to ada9d38 touches only `neuromri/p2_threshold_transport.py`,
   and removing the `__main__` joblib entry-point hunk reproduces the 469ee8f
   file exactly. The fix calls `run()` through the package module so that loky
   workers can unpickle fit functions; it changes no analysis line.
4. **Plan commit to production.** The same P2-relevant diff from the plan commit
   fc07982 to 469ee8f contains exactly the following non-analytic infrastructure
   hunks, each verified by exact text removal, and nothing else: the path-free
   run-identity hash (`config_hash` excludes machine paths; used for the
   recorded `config_sha256`/`config_hash` values only), SHA-verified node-local
   dataset staging (sourced only when `DATA_TAR` is set; the internal job reads
   cached embeddings, not images) and temporary-directory cleanup traps. These
   hunks were already part of the attested P1 production commit. The results
   plan is byte-identical at fc07982, 469ee8f, ada9d38 and the local checkout.
5. **Export identity.** Vega scope; SIF SHA-256 equal to the P1 SIF
   (`291ced61…`); `config_sha256` recomputed with the executed identity rule;
   configuration equal to `configs/p2.yaml` with the sbatch overrides
   `jobs=16`, `threads_per_fit=1`, `output=results/p2`; frozen image/role
   manifest hashes equal the local manifests; four recorded embedding-cache
   identities. Identity columns only: 442,800 metric rows forming the complete
   900-score-fit grid (4 encoders × 5 tasks × 5 folds × 3 seeds × 3 source
   clusters × 3 test clusters × 164 strategy/budget/draw rows), all with Vega
   scope and the provenance configuration hash; complete group and image
   prediction grids without duplicated unit-level keys; pip-freeze identical to
   the P1 Vega fits.
6. **Local analysis bytes.** Local CPU aggregation uses
   `neuromri/aggregate.py`, which differs from ada9d38 only by the already
   reviewed `--source-integrity-gate` local-provenance option.

## Attested, not recorded

`tracked_dirty_lines=0` was recorded once, when the attestation was collected
after the job ended; it is not a run-time record. Report clean-checkout status
as **attested (tracked files), not recorded at run time**. The embedding caches
and the SIF are identified by their recorded hashes; their bytes are not in the
export and were not rehashed.

The fallback accepts only the exact missing-git error in both fields after all
checks pass. Unknown errors, a dirty status, other commits, incomplete grids,
non-Vega scope, altered configuration or inputs remain rejected. No result row
is excluded, repaired or rewritten.
