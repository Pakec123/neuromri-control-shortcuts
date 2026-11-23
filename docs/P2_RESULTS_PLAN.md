<!-- Public sanitized rendition; site paths and obsolete operational references removed. Analysis specification unchanged. -->
# P2 manuscript results plan — iteration 3, 19 November 2026

Iteration-3 amendment, fixed before Zenodo score inspection: E4 adds real AD
positives. E3 remains negatives-only. All E4 numbers must be regenerated on Vega
with all four encoders and a complete results manifest. See the iteration-3
design for single-site provenance, unresolved source patient overlap, inherited
SPM standard-space full T1 at measured 1.5mm (the 1mm files are GM maps),
gender-code missingness and conditional uncertainty.

| Added/updated item | Exact content | CSV inputs | Interpretation |
|---|---|---|---|
| Table 1 extension | Zenodo AD/control ages, released gender-code counts, missing decoded sex, Alava recruitment/site, measured 1.5mm SPM processing and overlap limitation | `X/manifests/subjects.csv`, `E/cohort_demographics.csv`, `E/ad_roles.csv`, committed `zenodo3935636_provenance.json` | Descriptive; separate released numeric gender from decoded M/F |
| Table 6 | E4 C1 fold0 seed5101, four encoders: test AUROC/CI; sensitivity/specificity/BA and CIs at source BA/source specificity90 and target empirical k8; absolute-error delta/CI; raw/Holm p | `E/E4_metrics.csv`, `E/E4_tests.csv` | New conditional Holm family of 8: 4 discrimination + 4 specificity-change tests; no sensitivity-preservation test |
| Figure 1 update | Add AD source probes → Zenodo 9 control calibration / 35 AD + 10 control test; one reported external site | `E/ad_roles.csv`, `E/development_roles.csv` | All slices grouped by real external subject; source groups remain proxies |
| Figure 5 | E4 AUROC panel beside sensitivity/specificity versus budget for all four encoders, source operating points and .90 specificity reference | `E/E4_metrics.csv`, `E/E4_tests.csv` | Show the observed ranking/decision relationship, including failure to preserve AUROC or sensitivity; k16/k32 and tolerance bounds explicitly unavailable |
| Table S8 | Every E4 fold/seed/source cluster/strategy/k/draw/status and calibration identity, source model C/temperature | `E/E4_metrics.csv`, `E/predictions.csv`, `E/ad_roles.csv` | Repeated conditional evaluations, never pooled as people |

The historical negatives-only AD wording below applies **only to E3 and the
iteration-2 analysis**, not to the new E4. E4's local recruitment evidence
supports separate cohort origin from ADNI/OASIS, while unknown NeuroMRI source
identities still prevent certifying patient-level non-overlap. Restoring
specificity on 10 final controls cannot establish clinical utility. Report
sensitivity losses as prominently as specificity gains. The hypothesis of high
AUROC with wrong decisions must be evaluated honestly even if falsified.

---

# Historical iteration-2 plan (E4 additions above take precedence)

Target: Bioengineering special issue “Next-Generation Diagnostic and Therapy
Systems for Neurodegenerative Diseases”. The contribution is an audit of frozen
MRI decision thresholds under negative-cohort shift in AD/PD tasks, including
the limits of target-negative calibration. It is not a clinically validated
diagnostic system. Tumour/MS are supplementary specificity stress tests.

Freeze this amendment before the iteration-2/Vega outcomes. Iteration-1 local-development
results were already inspected; this is not a retrospective preregistration.
The age rule and diagnostic choices do not use scores. Final numbers must come
from the Vega main SIF, all four encoders, with a complete results manifest.

## Paths and reporting rules

`X` = the run directory printed by `bash tools/run_external.sh`;
`E` = `X/probes`. Every external CSV below is produced by the committed code.
`I` = the original internal P2 aggregation directory: run `neuromri.aggregate`
over the completed Vega P2 runs and use its `all_metrics.csv`, `summary.csv`,
`paired_difference.csv`, and `availability.csv`. Internal runs retain their own
SIF/input manifest. No local-development number enters manuscript result tables.

Use all encoders (ResNet18, ResNet50, DINOv2-S, BiomedCLIP); never select the best
encoder, seed or fold. Main conditional inference uses source fold 0, seed 5101,
cluster C1; repetitions are displayed as descriptive ranges in supplements.
AD C1 is the primary disease-matched internal control arm. PD source clusters
are descriptive stress analyses because no matched NeuroMRI control source
exists. E1 is estimation-first, not a confirmatory test of source confounding.
No pooling of slices, fits or draws as independent people. All external CIs are
conditional on fit; NeuroMRI CIs refer to proxies with unknown patient identity.

## Main manuscript

| Item | Exact content | CSV inputs | Status of numbers |
|---|---|---|---|
| Table 1 | Cohort/site/diagnosis n, age median [IQR], range and missing n, F/M/missing; source/modality/access and NeuroMRI unknown patient count | `E/cohort_demographics.csv`; `X/manifests/subjects.csv`; committed `manifests/external/archive_members.csv`; source `manifests/images.csv` | Descriptive; image counts cannot be called patient counts |
| Table 2 | Internal AD/PD: source-specificity90 versus target k=32 empirical quantile, specificity, sensitivity and BA at matched/mismatched negative clusters; availability of tolerance bounds | `I/all_metrics.csv`, `I/summary.csv`, `I/paired_difference.csv`, `I/availability.csv` | Original internal protocol's prespecified contrasts; no new external confirmatory claim |
| Table 3 | E1: internal proxy AUROC, external subject AUROC/95% CI and internal-minus-external AUROC loss; BA/sensitivity/specificity at source-BA and source-specificity90, both PD cohorts, four encoders | `E/E1_metrics.csv`, `E/E1_contrasts.csv` | Descriptive/estimation; modality/rendering/cohort/demographic shifts inseparable |
| Table 4 | E2: both directions, subject AUROC/95% CI, source-BA BA/sensitivity/specificity; fold-0 seed-5101 two-sided permutation p and Holm p | `E/E2_metrics.csv`, `E/E2_tests.csv` | Confirmatory conditional above-chance tests: 4 encoders × 2 directions, Holm family of 8; other metrics descriptive |
| Table 5 | Full IXI E3, AD C1: per-site source-specificity90 and k=32 empirical specificity, mean-over-draw absolute-error delta/subject-bootstrap CI, conditional specificity-change p/adjusted p; PD C1 panels separately labelled | `E/E3_metrics.csv`, `E/E3_contrasts.csv`, filter `analysis=full` | Prespecified primary E3 estimation; conditional Holm family 4 encoders × 2 tasks × 3 sites (24). PD panels descriptive stress analyses. p tests specificity change, **not** absolute-error improvement |
| Figure 1 | Data/decision pipeline: source train/validation/calibration, external PD test and separate IXI target-cal/test pools; show numbers and missing clinical AD positive arm | `X/manifests/subjects.csv`, `E/ixi_roles.csv`, `E/development_roles.csv`, committed `manifests/external/archive_members.csv` | Descriptive counts; no images redistributed |
| Figure 2 | Internal AD/PD specificity and sensitivity versus calibration budget, paired strategies; distinguish source-matched and shifted controls | `I/all_metrics.csv`, `I/summary.csv` | Original internal protocol; calibration draws are repeated evaluations |
| Figure 3 | E1 internal/external AUROC loss with conditional CIs plus E2 directional AUROCs; all four encoders | `E/E1_contrasts.csv`, `E/E2_metrics.csv` | E1 descriptive; E2 test annotations only from Table 4 family |
| Figure 4 | Full IXI specificity versus k, three sites × AD/PD; .90 reference, empirical and tolerance curves, explicit unavailable budgets, worst-site panel | `E/E3_metrics.csv`, filter `analysis=full`; `E/E3_contrasts.csv` | Primary E3 estimation; draw quantiles labelled as draw variation, not subject CIs; no sensitivity/AUROC axis |

## Supplementary material

| Item | Exact content | CSV inputs | Status of numbers |
|---|---|---|---|
| Table S1 | Complete external inventory, hashes, licensing/attribution, demographics including secondary IXI, actual source weights/SIF identity | `X/manifests/downloads.csv`, `X/manifests/subjects.csv`, `E/cohort_demographics.csv`, committed `manifests/external/archive_members.csv`; hash inventory `X/RESULTS_MANIFEST.json` | Descriptive/reproducibility; retain CC BY-NC-SA 4.0, non-commercial and no image redistribution |
| Table S2 | Every E1 source fold/seed/cluster/encoder and four source threshold strategies; no selected winner | `E/E1_metrics.csv`, `E/E1_contrasts.csv` | Descriptive conditional fits |
| Table S3 | Every E2 source fold/seed/direction and thresholds, LR C and target n; full development memberships | `E/E2_metrics.csv`, `E/development_roles.csv` | Descriptive; no additional significance family |
| Table S4 | All six E3 strategies, budgets, statuses and per-site/worst-site metrics, all source clusters and fits | `E/E3_metrics.csv`, `E/E3_contrasts.csv` | `full` primary; repetitions/PD clusters descriptive; unavailable values missing |
| Table S5 | Score-blind age-matched IXI: site/role counts, age/sex, matching gaps/unmatched PD references; same E3 endpoints at feasible k and paired k32 contrasts where supported | `E/ixi_age_matched_roles.csv`, `E/cohort_demographics.csv`, `E/ixi_roles.csv`, `E/E3_metrics.csv`, `E/E3_contrasts.csv`, filter `analysis=age_matched`; `X/manifests/subjects.csv` | Secondary descriptive; no Holm p assigned to age-matched rows; IOP k16/k32 unavailable |
| Table S6 | PD-only domain classification: five-fold out-of-fold AUROC, source proxy/external subject counts; 32-bin foreground JS and Wasserstein distances, including NEUROCON–Tao Wu same-modality comparison | `E/diagnostics/domain_metrics.csv`, `E/diagnostics/domain_predictions.csv`, `E/diagnostics/intensity_distances.csv` | Descriptive joint shift diagnostics; cannot partition modality from source effects |
| Figure S1 | Histogram profiles for NeuroMRI PD and each external PD cohort, one equal-weight histogram per proxy/subject; external–external comparator | `E/diagnostics/intensity_histograms.csv`, `E/diagnostics/intensity_distances.csv` | Descriptive rendering/intensity comparison, not a sequence classifier |
| Figure S2 | Full versus age-matched IXI age distributions and E3 specificity/absolute-error budget curves; mark incomplete age support and IOP sample limits | `X/manifests/subjects.csv`, `E/ixi_age_matched_roles.csv`, `E/ixi_roles.csv`, `E/E3_metrics.csv` | Secondary descriptive; no claim that sex or other covariates were matched |
| Figure S3 | Calibration draw variation and tolerance-bound availability; per-site error-delta distributions across source fits | `E/E3_metrics.csv`, `E/E3_contrasts.csv` | Descriptive uncertainty; do not bootstrap draws as people |
| Table S7 | Internal Tumour/MS transport, original strategies/budgets; source proxy support and exclusions | `I/all_metrics.csv`, `I/summary.csv`, `I/availability.csv` | Secondary stress analyses |

## Manuscript-facing boundaries

“In two independent T1 PD cohorts, we evaluated discrimination and source-threshold
transport of frozen probes. NeuroMRI PD has T2-like visual contrast, with no
verified sequence labels. The released external archives contain only T1
MPRAGE anatomy and BOLD fMRI. E1 therefore measures joint acquisition, rendering,
cohort and demographic shift; domain separability does not identify the causal
contribution of source confounding or modality.”

“The present external AD arm evaluates false-positive control on healthy IXI
volunteers only. External AD AUROC, sensitivity, balanced accuracy and clinical
benefit are not identifiable. Restoring specificity does not establish preserved
sensitivity. OASIS/ADNI/MIRIAD access remains DUA-blocked. A renewed bounded search
identified public CC BY AD/MCI candidates, recorded in the private historical search notes (omitted from this release);
these have not been integrated into the frozen positive-cohort protocol. Their
availability means that AD-positive data are not universally inaccessible.”

No image-based montage is a planned distributable artifact. Any later figure
containing cohort images needs a separate licence-compliant publication decision.
