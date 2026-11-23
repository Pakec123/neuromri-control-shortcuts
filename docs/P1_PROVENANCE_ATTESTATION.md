<!-- Public sanitized rendition; site paths and obsolete operational references removed. Analysis specification unchanged. -->
# P1 integrity-gate amendment

Dated 20 November 2026. Written AFTER final results export but BEFORE any
scientific endpoint was computed or inspected in the P1 manuscript workflow.
Iteration 3 inspected identities, schemas and artifact hashes only and stopped.
This is a post-export acceptance-rule amendment, not a change to the version-2
analysis plan, estimands, tests, families, seeds, draws or training protocol.

The production compute nodes and original SIF lack git. All 1350 fit records
retain the original missing-git error for both commit and dirty status. The
future provenance fix in main commit 808ca5e does not apply retroactively.

For this export only, executed commit identity may instead be established by
all of the following machine checks in `tools/p1_integrity_gate.py`, implemented
by `tools/p1_attestation.py`:

- The six extracted attestation files equal the bytes in the independently
  collected tar, whose SHA-256 is
  `aa1d1ec1bff803852a3cffa1c3dfc6766fac608df75614171042b3e4faf4c345`.
- The production commit/tree record matches local git; the reverse-chronological
  reflog places HEAD at 469ee8f1aa6f5bbacaf6dc3d078b9920a01b5941 from
  19 November 19:10:08 +01:00 until the adjacent next HEAD ada9d38 at
  20 November 08:26:14 +01:00.
- Exactly 338 distinct expected array tasks (50548178_0 and 50548180_1..337)
  are COMPLETED with ExitCode 0:0, nonempty nodes and the single production
  WorkDir `<ORIGINAL_INTERNAL_WORKDIR>`. Every submit,
  start and end lies strictly inside that reflog window, in Europe/Ljubljana
  time; Submit <= Start <= End. The maximum End equals `last_end.txt`.
- The supplied P1-relevant diff is empty. An independent local git comparison
  of the two commits verifies this across P1 training dependencies, configs,
  slurm, env, manifests and weight registry. Those local training/input paths
  must also match production. The complete supplied diff is retained and hashed.
- Every fit's original numeric SLURM identity is present; adjacent four-row
  runlist blocks have a single identity (last block has two fits), with 338
  distinct identities and the expected first/last array-parent numeric IDs.
  Array notation is exported by sacct; its raw numeric JobID mapping is not.
  Assignment of intermediate numeric IDs to array indices is inferred from
  the committed four-row dispatcher, not independently observed in this export.

Claude's read-only operational record, supplied by the user on 20 November,
attests that immediately before the 08:26 fast-forward,
`git status --porcelain --untracked-files=no` returned zero lines; the clone
was frozen throughout P1 under the rule "do not pull while P1 runs" and nobody
edited it. Ignored `slurm/hpc.local.env` contains account/partition/paths only.
This operational statement is not a per-fit machine-recorded clean status.
Report it as **attested (tracked files), not recorded per fit**. It supplements
the scheduler/reflog checks and must not be represented as new runtime evidence.

The fallback accepts only the exact missing-git error in both original fields,
after the complete attestation passes. Unknown errors, dirty status, other
commits, incomplete exports, altered protocol/config/input bytes, non-Vega
scope and hardware/container failures remain rejected. No original provenance
is rewritten, no failed fit is excluded, and all other gate checks remain.

The original 1400-file statement remains an uncorroborated supplied count:
Iteration 3 enumerated exactly 1350, one for every planned run, with no extra 50.
No extra file is deduplicated or selected. This amendment changes only the
commit/checkout evidence route and retains that inventory finding.
