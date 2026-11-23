<!-- Public sanitized rendition; site paths and obsolete operational references removed. Analysis specification unchanged. -->
# P1 primary analysis plan

Version 2, fixed 20 November 2026 before inspection of any final Vega result.
This version supersedes the descriptive-only reporting choice in the 19–20
November draft. It is a versioned prespecification, not an external registration.
No final result, pilot performance estimate or architecture ranking was used to
choose the tests. Amendments after this date must retain this text in git history,
state their date and reason, and identify whether results had been inspected.

## Design and estimands

The main design is 810 core plus 540 repair fits: six backbones, AD/Tumour/MS,
core alpha 0/0.5/1, five folds and seeds 5101/5102/5103. Repairs cover ResNet50,
ConvNeXt-T and ViT-B/16 with four policies on identical selected support.
Training remains unchanged, with maximum 30 epochs and early stopping patience 5.
Proxy groups, rather than images or recovered patients, are evaluation units.
The fixed decision threshold is 0.5 (score >= 0.5 is positive).

For each of the 18 task by backbone combinations, the primary estimand is
Delta-alpha = AUROC(alpha=1) - AUROC(alpha=0), on the appearance-matched final
test stratum: AD C1, Tumour C2, MS C0. Pair on identical held-out group IDs,
labels and stratum assignments within fold by seed. Average the 15 paired
fit differences with equal weights. Secondary core estimands use the same
contrast for matched-stratum balanced accuracy and worst-stratum specificity.
The latter is the minimum specificity over C0/C1/C2 within each fit, not the
specificity of only the matched stratum. It is minimized again in every resample.
Alpha 0.5 and the remaining endpoints are descriptive.

The primary repair contrast is equal_negative_cluster minus pooled_erm on
worst-stratum specificity, separately for each of 3 backbones by 3 tasks.
Pair within fold by seed on common support and final groups; average 15
differences. Other repair policies and endpoints remain descriptive. Standardized
endpoints average stratum metrics, not concatenated scores.

## Uncertainty and tests

Use 2000 two-level bootstrap resamples and NumPy PCG64 seed 6101. Each resample
first samples 15 fold by seed fits with replacement. For every selected
occurrence, independently resample that fit's final proxy groups with replacement
within positives and within each negative stratum, retaining original cell sizes.
Reuse the identical group draw for both scorers; shared positives are sampled
once per occurrence and reused across strata. Recompute metrics, differences,
and their equal-fit mean. Report 2.5th and 97.5th percentiles (NumPy linear
quantiles) as pointwise 95% CIs. Resampling fits includes observed training
variability; it is not a new set of training refits. No prediction pooling or
image bootstrap is permitted. Independent deterministic streams for each
contrast and for bootstrap/permutation derive from SeedSequence(6101), with
core backbone order resnet50, convnext_tiny, efficientnet_v2_s, vit_b_16, swin_t,
dinov2_b, tasks AD, Tumour, MS, followed by repair backbones resnet50,
convnext_tiny, vit_b_16 in the same task order. Fits sort by fold then seed.

For each primary test use the absolute mean paired fit-level difference as the
statistic. Generate 2000 independent Rademacher sign vectors across the 15 fits,
without enumerating or excluding any vector. The two-sided Monte Carlo p value
is (1 + count(abs(permuted mean) >= abs(observed mean))) / 2001, counting ties.
Comparisons allow an absolute tolerance of 1e-14 for floating-point sum ties.
Apply Holm step-down adjustment across all 18 primary core AUROC tests, and
separately across the 9 primary repair worst-specificity tests. The family-wise
level is 0.05 for each family; there is no combined 27-test error guarantee.
Report raw and adjusted p values, all effects and CIs including null findings.
Secondary endpoints have pointwise intervals but no inferential p values.
Pointwise CIs are not multiplicity-adjusted and need not agree with Holm decisions.

Overlapping training sets and repeated test groups across seeds induce dependence.
Fit resampling and fit-level sign flips approximate the repeated-fit experiment;
the sign test assumes joint invariance to the sampled sign flips under the null.
Holm controls multiplicity only insofar as its input p values are valid. Neither
procedure removes dependence, establishes patient-level population inference,
or accounts for uncertainty in the forensic grouping. State these limitations.

As sensitivity, fit Delta ~ 1 + (1|fold) + (1|seed) separately for each primary
contrast using REML in statsmodels if present in the pinned environment. Use
crossed variance components (one overall group, re_formula='0', vc_formula
fold='0 + C(fold)', seed='0 + C(seed)'), L-BFGS with maximum 1000 iterations.
Report intercept, Wald 95% CI and convergence/warnings; a failed or singular
fit remains documented, without a replacement model. It is descriptive sensitivity,
not a second family of significance tests. At prespecification, statsmodels is
absent from env/requirements.lock and from the local CPU environment. Therefore
the current locked-environment analysis records this sensitivity as unavailable;
do not install into or alter the frozen training environment to obtain it.

## Existing outputs and failure rules

neuromri/p1_finetune.py already persists predictions.csv, one row per group,
with group_id, y, normal_cluster, score and n_images. Scores are image-probability
means. metrics.csv has encoder/task/alpha/fold/seed/policy/test_cluster and
endpoints, including worst_negative_specificity. No training change is required.
The CPU-only neuromri.p1_stats module reads these files and the committed runlist,
checks unique paired IDs and metadata and verifies recomputed endpoints against
the raw metrics. Reject missing fit pairs, nonfinite/invalid scores or endpoints,
duplicate identities, inconsistent labels/strata and absent evaluation strata.
Require all five folds by three seeds for every contrast; never drop failed fits.

If group predictions were genuinely not persisted, explicitly report their
absence and use fit-level resampling only, with the CI labelled accordingly.
This fallback is allowed only when predictions.csv is absent from every
contributing fit; reject partially missing exports and malformed predictions.
It omits test-group sampling uncertainty and cannot independently verify pairing.
It is unnecessary for the currently verified runner. Do not modify running training.

The module writes analysis.json, estimates.csv and fit_deltas.csv, recording
input hashes, plan/module/lock hashes, package versions, seed, draw count,
bootstrap scope, sensitivity status and all 15 differences per contrast.
The manuscript reader checks these identities and consumes this output for its
primary tables, forest plot and abstract inference placeholders. The existing
45 fixed-fit conditional contrast outputs remain supplementary diagnostics.
All numerical displays stay [VEGA] until the full final export passes its scope,
provenance, support and completeness gates. Synthetic tests use temporary folders
and never populate the delivered draft.

## Architecture and reporting

Show a forest plot of the 18 Delta-alpha AUROC estimates and pointwise CIs,
grouped by task and labelled by backbone. Describe differences between backbone
effect estimates without an omnibus test or a superiority claim from CI overlap.
Report mean/minimum/maximum endpoints over fits separately from inferential CIs.
Main Table 6 gives the 9 primary repair contrasts, CI and Holm p value;
Table S1 gives the 18 core effects, CI and Holm p value plus secondary effects.
Abstract: one primary-inference sentence with a selected, prespecified ResNet50
AD effect/CI and the count of Holm rejections across 18 tests. Report all 18
effects in the figure/table regardless of that count. Mixed-model results and
all fit deltas accompany the machine-readable supplementary outputs.

Supplementary File S1 is an exact copy of this file. Source identities are
captured in analysis.json and the manuscript audit, without changing this dated
plan when artifacts are rebuilt. No external patient experiment or strict-grouping
refit is added to the 1350-fit design.
