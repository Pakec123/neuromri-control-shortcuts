# Public snapshot provenance and release adaptations

The associated manuscript is under review and is not included. This repository
has fresh history and contains no private development objects or remote URL.

The baseline is development commit `75eff81d1904a6605109414e2896d7d850e6df77`.
For all retained development paths, a git diff against `ada9d38` found only:
`neuromri/common.py` (file-based commit lookup/source fingerprint provenance),
three SLURM dispatchers (explicit per-step memory), the new
`slurm/four_step_memory.sh`, and the provenance/fanout regression tests. Against
`808ca5e`, only those SLURM memory changes and the new fanout regression test remain. The analysis modules,
configuration values, run list, weight registry and frozen input manifests have
no numerical-method differences from either result-producing commit.
The joblib entry-point fix was already present at `ada9d38` and `808ca5e`.
P1 actually ran at `469ee8f`; its training code/input bytes match `ada9d38`,
which is the internal P1/P2 reporting reference. Do not claim it recorded that
later commit per fit.

Release-only changes beyond the baseline are:

- Add the later dated P1/P2 attestation gates, CPU reanalysis helpers and tests
  from manuscript branch commit `cdf7e8b`, because they are absent from the
  development main branch. Extract only the unchanged identity/provenance/CSV
  comparison functions into `neuromri/result_validation.py`; no Word builder,
  manuscript prose or journal template is included.
- Add the manuscript branch's gated-local-provenance option to `aggregate.py`.
  The numerical aggregation/bootstrap functions remain unchanged. The same
  Part A command generator is relocated to `tools/p1_analysis_commands.py`.
- Rename the site settings to `hpc.env` / `hpc.local.env`, rename the required
  external launcher to `tools/run_external.sh`, remove its fixed partition
  restriction, and replace site paths/defaults by documented placeholders.
  The literal `scope: vega` is retained for compatibility with recorded identities.
- Redact original WorkDir strings from public attestation docs and gates.
  Supply the exact original value through `NEUROMRI_ORIGINAL_INTERNAL_WORKDIR`
  or `NEUROMRI_ORIGINAL_EXTERNAL_WORKDIR` when auditing original private evidence.
  Supply `NEUROMRI_ATTESTATION_REPO` for read-only access to the original object
  database. Those historical gates require private scheduler/reflog exports and
  original git objects; they cannot certify historical jobs from this fresh
  repository alone. They fail closed when that evidence is absent. Public docs
  are sanitized renditions, not the byte-identical embedded journal appendices.
  Source-identity comparisons against the original paths also require the
  original audit checkout; the curated checkout is intentionally different.
- Retain all portable numerical/scheduler rejection tests; mock historical git
  responses in the P1 unit fixture and omit the integration test that requires
  the private history. No authentication or historical source check is bypassed
  in production code.
- Add README, citation metadata, portable environment template and reproduction
  drivers. Original dataset filenames, subject identifiers, released age/sex,
  folds and role assignments are retained; no images/clinical archives are bundled.

Fresh runs record their own public commit and source fingerprint; use their
complete manifests and standard runtime provenance. Never label a fresh run as
one of the historical development commits. The old amendment gates document and
validate the specific original exports, not arbitrary new experiments.

`build_manifests.py` is retained as the original freeze utility. Reproduction
uses the committed manifests; its exploratory audit inputs are not distributed.
Do not regenerate proxy groups/folds or substitute new identities for this paper.

No licence is granted by this snapshot pending the PI's licence decision.
Dataset and pretrained-weight terms are separate and remain with their publishers.
