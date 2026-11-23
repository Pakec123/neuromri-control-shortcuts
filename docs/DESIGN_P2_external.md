<!-- Public sanitized rendition; site paths and obsolete operational references removed. Analysis specification unchanged. -->
# P2 external validation — iteration 3 amendment (19 November 2026)

Fixed before viewing any Zenodo probe outcome. E1–E3 remain unchanged. Add **E4:
AD external discrimination and threshold transport**, all four encoders and the
same NeuroMRI AD source fits (no new tuning on external outcomes). C1 is primary;
other source clusters/folds/seeds remain descriptive. High AUROC with poor source
decisions is a testable hypothesis, not an inclusion criterion or promised result.

## Zenodo 3935636: provenance, labels and preprocessing

[Publisher record](https://zenodo.org/records/3935636) and
[API](https://zenodo.org/api/records/3935636) identify CC BY 4.0, `ea`=AD,
`crl`=healthy control and `tb`=bipolar disorder. Download only `T1_preprocessed.zip`
and `Clinical_data.zip`; keep raw immutable, commit byte counts/SHA256s. Join
diagnostic prefix and numeric anonymous ID to `clinical_data_id_age_gender.csv`,
reject duplicates, mismatches and missing clinical records; exclude all `tb`.
Released age and numeric gender code are retained. Gender code semantics are not
documented in that CSV/record: do not silently infer M/F; sex remains missing.
Released controls include two age-56 records despite the linked publications'
older-than-64 study description. Preserve released ages, flag this discrepancy,
and do not silently correct ages or exclude subjects after inspecting scores.

The linked [cohort publication](https://doi.org/10.3389/fnagi.2017.00179)
reports local recruitment at Alava University Hospital, Vitoria, Spain, and a
1.5T Siemens Magnetom Avanto acquisition. This supports independent cohort origin,
**not ADNI/OASIS derivation**. The scanner statement describes the study's DWI;
T1-specific TR/TE/flip angle are not supplied by the Zenodo record. No per-subject
scanner/site sidecars are supplied. Assign one reported cohort site, Alava;
do not invent acquisition parameters or site-held-out analyses within it.
NeuroMRI's reported OASIS-derived Kaggle ancestry and missing patient identifiers
prevent certifying individual non-overlap. No independence claim rests on
different filenames or unequal image hashes. If contrary derivation evidence
emerges, E4 becomes a non-independent transport sensitivity analysis and is
excluded from independent external-validation claims.

**Measured archive correction to the publisher description:** ZIP central-directory
inspection and NIfTI headers show 78 original T1s, 78 warped full-T1 images under
`T1_preprocessed_SPM_1.5mm/T1_brain/wmrf<diagnosis>_<id>.nii`, and segmented/smoothed
tissue derivatives. The **1mm DARTEL files are smoothed grey-matter maps**, not
full anatomical T1. Select only the warped bias-corrected **1.5mm full T1** images
(121×145×121, axis-aligned, common affine; sample header description "Warped bias
corrected image"). Do not use the 1mm GM maps as T1 or treat native originals as
MNI-registered. This discrepancy was resolved before any E4 scores were viewed.

These selected images already occupy the released standard-space grid. Validate
axis-aligned **1.5mm** geometry after lossless canonical RAS reorientation; **skip
image/mask world-grid resampling and any second registration**. Keep pinned
SynthStrip foreground masking with CSF retained, the same five 40/45/50/55/60%
mask-extent planes, subject-wide 1st/99th intensity scaling and RGB224 aspect
padding. Convert the 5mm crop margin to ceil(5/1.5)=4 voxels (6mm) to avoid
interpolation. SynthStrip internally resamples for mask estimation; its output
mask returns to the released input geometry. The sliced T1 is never warped or
interpolated spatially by our 3D pipeline. Inherited upstream warping, bias
correction/brain processing and 1.5mm spacing differ from the other cohorts'
world-1mm resampling without atlas registration; they remain part of the transport
estimand. Do not claim identical rendering pipelines or recover removed tissue.
Every slice records which spatial branch was used.

## E4 split, endpoints and inference

Freeze seed 7201, sorted subject IDs, then shuffle controls within reported site;
floor(n_controls/2) form the target-negative calibration pool, remaining controls
and **all AD cases** form the untouched test set. For 19 controls: 9 calibration,
10 test controls; 35 released AD cases test-only. Publish `ad_roles.csv`. Never
fit a disease probe, C, temperature or image processing parameter on this cohort.
Only target quantiles use calibration controls; all slices of each person stay
together. All external subjects/site are absent from source fitting, subject to
the source ancestry limitation above. A single released site cannot support a
within-cohort held-out-site contrast.

Report test subject AUROC and 95% stratified subject-bootstrap CI; sensitivity,
specificity and BA with subject CIs at fixed .5, source BA, source specificity90,
source temperature .5, and target empirical/tolerance quantiles. Keep original
k={0,4,8,16,32} ×20 draws. k16/k32 exceed the pool; all positive-k tolerance
bounds are unavailable (need >=29 controls). Mark missing explicitly; no smaller
budget substitution. k0 uses source specificity90. Each target draw is shared
across encoders/source models. Scores average five slice probabilities/person.
Discrimination is evaluated on the same AD/control test set for all thresholds.

One new **Holm family of 8**: four encoder-specific two-sided subject-label
permutation tests of |AUROC−.5|, plus four paired subject-sign tests of specificity
change, target empirical **k8** versus source specificity90, averaging the 20
draws. Only source fold0/seed5101/C1 enters this family. k8 is fixed from pool
capacity before scores. Report paired absolute-specificity-error difference
and subject-bootstrap CI separately: a specificity-change p value is not an
error-improvement test. Also report paired mean-over-draw sensitivity and BA
changes with stratified subject-bootstrap CIs. Report endpoints at both thresholds
without a preservation/noninferiority claim. CIs condition on model/calibration
pool/draw; 20 draws, folds and seeds are never independent people. 2,000 bootstrap
and permutation draws on Vega, 200 for optional local smoke. Brier and AUPRC use constructed
test prevalence and remain descriptive. No threshold selection by test outcomes.

figshare 30850301 is deferred optional MCI/control; MCI is never relabelled AD.
OASIS/ADNI/MIRIAD registration/DUA blocks remain. Credit Besga, Graña and Chyzhyk,
Zenodo DOI 10.5281/zenodo.3935636, CC BY 4.0; no images redistributed.

---

# Historical iteration 2 amendment (AD availability statements superseded above)

Amendment fixed 19 November 2026 before iteration-2/Vega outcomes. Iteration-1
smoke outcomes were known at review; this is not a retrospective preregistration.
All four encoders remain in the grid. Both source and external features must be
recomputed with the Vega main SIF. See `RUN_P2_external.md` for the single-command
chain and `P2_RESULTS_PLAN.md` for exact manuscript CSV/table/figure mappings.

## Modality audit and descriptive diagnostics

Exhaustive archive inspection: NEUROCON has 43 T1, 85 BOLD, 128 JSON and one TSV;
Tao Wu has 40 T1, 40 BOLD, 80 JSON and one TSV. All 83 anatomical sidecars specify
MPRAGE (IR method). All “other” files are metadata. `archive_members.csv` and
`archive_modalities.json` record every member; reproduce with
`tools/inspect_external_archives.py`. No T2/PD-weighted/FLAIR anatomical E1
variant is available. BOLD must not be substituted for anatomical T2.

Add two score-independent diagnostics on PD images only:

* Domain classifier: compare NeuroMRI PD separately with NEUROCON PD and Tao Wu
  PD. Average L2-normalized frozen slice features within each source proxy or
  external subject. Fixed five-fold stratified CV of these units, seed 9101,
  training-only StandardScaler, class-balanced LR C=.1, no tuning. Report
  out-of-fold domain AUROC and unit counts for all four encoders. All slices of
  a unit stay together; source patient identity remains unknown.
* Foreground intensity: uint8 luminance, exclude exact zero background, 32 equal
  bins over [0,256). Normalize each image histogram, average images per unit,
  then units equally per domain. Report base-2 Jensen–Shannon distance (square
  root of divergence) and Wasserstein distance on [0,1] bin centres. Include
  NEUROCON PD versus Tao Wu PD as a common-T1 comparator. These describe the
  actual rendered inputs, not calibrated MRI signal or verified modality labels.

These diagnostics quantify **joint** modality/rendering/site/demographic/cohort
shift. They cannot partition causal contributions or prove source confounding.
They are descriptive, receive no significance tests, and do not trigger retuning.
Outputs: `probes/diagnostics/{domain_metrics,domain_predictions,intensity_histograms,intensity_distances}.csv`.

## Score-blind secondary age-matched E3

Full IXI remains primary. Reference: pooled 47 PD cases from NEUROCON/Tao Wu
with known ages. Preserve original IXI site-specific target-cal/test roles
(seed 7101). Separately within each site and role, match without replacement,
one IXI person per reference case, absolute age gap <=5 years. Maximize match
cardinality, then minimize total absolute age gap by linear assignment; sort
subject IDs for deterministic ties. Exclude missing ages only from this
secondary analysis. No sex, features, scores or observed outcomes enter matching.

Freeze `ixi_age_matched_roles.csv`, including reference IDs/ages and age gaps:
Guys cal/test 47/46, HH 43/41, IOP 10/12 (199 total). Report unmatched reference
cases and residual age/sex differences; this is not perfect demographic balance.
`cohort_demographics.csv` gives full cohort/site/diagnosis and matched IXI age,
sex and missingness. Sex is reported, not matched. The same fitted probes,
preprocessing, thresholds, seeds, budgets and draws are reused. No additional
disease-model fit is needed. For AD this PD-age reference is a common ageing
stress analysis, not a match to AD patient ages. IOP k16/k32 is explicitly
unavailable. Secondary rows have `analysis=age_matched`, primary rows `full`.
Only full-IXI rows receive the original conditional Holm tests; secondary
comparisons are descriptive.

## AD access update and attribution

OASIS/ADNI/MIRIAD remain **BLOCKED — needs PI**. The renewed search found public
CC BY 4.0 candidates: Zenodo 3935636 (AD/control T1, already nonlinearly registered
to MNI) and figshare 30850301 (MCI/control, skull-stripped T1). Thus AD-positive
data are not universally DUA-blocked. These candidates have not been integrated
into the frozen grid and need a prospective integration/QC/provenance amendment.
The current external AD arm is **E3 negatives-only**; external AD AUROC,
sensitivity, BA, preserved sensitivity and clinical benefit are not identifiable.
Historical AD candidate-search notes are not needed for this frozen E4 reproduction and are omitted. No registration/contact/DUA.

NEUROCON/Tao Wu attribution: FCP/INDI Parkinson's release,
https://fcon_1000.projects.nitrc.org/indi/retro/parkinsons.html; Badea et al. 2017,
https://doi.org/10.1371/journal.pone.0188196. **CC BY-NC-SA 4.0, non-commercial;
we do not redistribute images.** IXI retains CC BY-SA 3.0 attribution to
https://brain-development.org/ixi-dataset/.

---

# Iteration 1 protocol (historical access conclusions superseded above)

Design fixed on 19 November 2026, before external probe results. Development
results on the local GPU are smoke evidence; final manuscript results must be
regenerated on Vega. AD and PD are primary; existing Tumour/MS analyses remain
secondary. This arm does not change the internal P2 protocol.

## Data and access

* [NEUROCON and Tao Wu](https://fcon_1000.projects.nitrc.org/indi/retro/parkinsons.html):
  respectively 27 PD/16 controls and 20 PD/20 controls; both T1 MPRAGE and resting
  fMRI; 1.5T Siemens Avanto versus 3T Siemens Trio. The linked usage licence is
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
  Cite Badea et al. 2017, [doi:10.1371/journal.pone.0188196](https://doi.org/10.1371/journal.pone.0188196).
  Only anatomical volumes and metadata are extracted; functional scans stay in
  the unmodified downloaded archives. Released subject identifiers encode
  control/patient status; released `*_patients.tsv` files provide age/sex for
  cases and controls (their IDs omit the BIDS `sub-` prefix). Metadata
  inconsistencies cause an error rather than an inferred label.
* [IXI](https://brain-development.org/ixi-dataset/): nearly 600 healthy volunteers
  from Hammersmith (HH, Philips 3T), Guy's (Guys, Philips 1.5T) and Institute of
  Psychiatry (IOP, GE 1.5T), under CC BY-SA 3.0. Download T1 and demographics;
  defer T2 because the primary common external modality is T1. Site is explicit
  in released filenames, independently of the demographic spreadsheet. Exact
  available counts are measured from the archive, not assumed to be 600.
  These volunteers are negative-cohort controls, not biomarker-confirmed absence
  of AD/PD. IXI's age distribution is not matched to a clinical case population.
* [OASIS access](https://sites.wustl.edu/oasisbrains/home/access/),
  [ADNI access](https://adni.loni.usc.edu/data-samples/adni-data/) and
  [MIRIAD](https://www.ucl.ac.uk/brain-sciences/ion/research/research-centres/dementia-research-centre/research-clinical-trials/minimal-interval-resonance-imaging-alzheimers-disease-miriad)
  require registration/application and/or DUA acceptance: **BLOCKED — needs PI**.
  Exact requirements are in `manifests/external/access.json`. No registration,
  signature, login or contact is attempted. No independently sourced open AD
  positive cohort satisfying the no-registration/no-DUA constraint was verified
  in this bounded search. Historical direct OASIS links and Kaggle mirrors do
  not supply independent, currently authorised validation of OASIS-derived AD.
  Additional screening in `ad_screening.json`: OpenNeuro ds006148 has public CC0
  imaging but explicitly recruits cognitively normal participants (including
  amyloid-positive people), not clinical AD cases; its demographics need a DUA.
  OpenNeuro ds004767 provides CC0 ex-vivo AD-pathology MRI, outside this in-vivo
  whole-brain T1 estimand. These resources must not be relabelled as eligible
  clinical positives merely to remove the AD validation gap.

Raw lives at `<EXTERNAL_RAW_ROOT>/<cohort>/raw` on a local development machine. Download and
volume SHA256s, relative paths, source URLs, licences, diagnosis, subject IDs,
age/sex where available and geometry are committed under `manifests/external/`.
PNGs, masks, tool weights, caches and result tables are never committed.
Retain source attribution and the respective share-alike/noncommercial terms
with derived data; combined analyses do not relicense the original cohorts.

## Preprocessing and interpretation limits

Direct visual inspection of six deterministic examples each of NeuroMRI AD Mild,
PD and Normal (`results/t1_inspect.png`) found 224×224 RGB inputs. AD appears
axial T1 with isolated brain foreground; PD appears axial with T2-like bright
CSF and isolated brain foreground. These are appearance observations, not
sequence labels verified from DICOM. Normal mixes T1/T2/FLAIR-like contrast,
axial/sagittal/coronal views and stripped/unstripped coverage. Black background
alone does not establish skull stripping. Native orientation, slice location,
acquisition and patient IDs cannot be recovered from the 2D release.
Consequently ancestral patient overlap between NeuroMRI and an external release
cannot be certified absent from these identifiers alone. E2 has genuine
cohort-specific subject identities; E1 source-side patient independence remains
unresolved even though external subjects and their slices are grouped correctly.

Use pinned [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/)
(Hoopes et al., [2022](https://doi.org/10.1016/j.neuroimage.2022.119474)) model 1,
1mm brain border, retain CSF, for external T1 foreground isolation. Script,
model and surfa revision are pinned in `tools.json`; the only upstream code
patch disables cuDNN benchmarking. Seed 5101, deterministic torch algorithms,
TF32/flash attention disabled and CUBLAS workspace recorded. A separate venv
overlay holds pinned added dependencies; the frozen P1/P2 conda env is untouched.

The deterministic simpler alternative to atlas registration is:

1. Validate finite 3D volume and mask geometry. Canonical RAS+ orientation with
   nibabel 5.3.2; resample to world-axis-aligned 1mm voxels (linear image,
   nearest-neighbour mask). This removes voxel obliquity, not anatomical tilt.
2. Find the subject's brain-mask bounds. Select five planes at 40/45/50/55/60%
   of its inferior–superior extent, with no label-dependent choice. Record
   resampled slice index and world z. This targets central axial brain anatomy,
   spanning temporal/ventricular levels, without claiming MNI equivalence.
3. Mask background, normalize brain intensities using whole-volume 1st/99th
   percentiles, clip and round to uint8. One subject-wide in-plane mask bounding
   box plus 5mm margin avoids per-slice zoom. Anterior at top, subject right at
   right (neurological display), aspect-preserving bilinear padding to RGB224.
4. Save five PNGs/subject with hashes of volume, mask and pipeline. Arithmetic
   mean of slice disease probabilities is the subject score. Fit weights balance
   subjects and classes so subjects with more slices cannot dominate.

No parameter is optimized using diagnosis performance. A side-by-side montage
must be inspected before interpreting smoke results. This pipeline does not
harmonize sequence, head tilt, anatomical coordinates or original rendering.
In particular E1 T1-to-apparently-T2 PD transport combines modality, rendering,
cohort and demographic shifts: failure alone cannot isolate source confounding
or show absence of structural PD signal. No clinical diagnostic claim follows.
Atlas alignment is a future sensitivity analysis. The iteration-2 audit confirms no matched external T2 anatomy in these archives.

## E1 — NeuroMRI PD reality check

Frozen ResNet18, ResNet50, DINOv2-S and BiomedCLIP, with exactly the existing P2
weight manifest and image transforms. For each source fold/seed/control cluster,
train PD(K2) versus Normal(K1). Use existing proxy-group roles; balance training
negative group count across control-cluster arms, retaining identical positives.
L2-normalize features, fit StandardScaler on training only, choose LR
C∈{.01,.1,1} by group-mean source-validation log loss. Separate source-calibration
groups select thresholds/temperature. Keep source patient identity explicitly
**unknown**: proxy grouping is not subject-level separation.

Test every NEUROCON and Tao Wu subject without adaptation. Primary endpoint:
subject AUROC per external cohort and source model. Secondary: BA, sensitivity,
specificity at source BA and source-specificity90 thresholds, fixed .5 and
temperature .5, plus Brier/AUPRC at constructed prevalence. Also retain internal
held-out P2 scores as context with their different proxy-group estimand.
Hypothesis: strong internal performance will not transport (descriptive
prespecified meaningful AUROC loss ≥.10). Falsification: consistently preserved
ranking and decisions on both external cohorts; a failure is not presumed.
An independent-cohort stratified bootstrap of internal/external test units can
estimate the AUROC loss, but does not repair unknown NeuroMRI patient overlap.

## E2 — PD cross-cohort signal

NEUROCON→Tao Wu and Tao Wu→NEUROCON. Five stratified folds of **source subjects**:
one source fold reserved for calibration, remaining source subjects split
stratified 75/25 into training/validation. Choose LR C on validation subject
log loss; source-calibration fold alone sets thresholds. Repeat the full scheme
for seeds 5101/5102/5103. Target cohort never tunes a model or threshold. Each
model predicts all target subjects; report conditional fold/seed results and
their range. Repeated target predictions are not new subjects. A prespecified
fold ensemble may be reported descriptively only, with distinct identification.

Primary: external subject AUROC in each direction; secondary BA/sensitivity/
specificity at source thresholds. Hypothesis: some structural signal may survive
after removing perfect source/label confounding. Two-sided subject-label
permutation tests (2,000 permutations, conditional fixed target scores),
Holm-adjusted over 4 encoders×2 directions. Falsification of above-chance signal:
no reproducible gain over .5, broad intervals including chance, or opposite
directions; nonsignificance is not evidence of equivalence or no signal.

## E3 — external negative-site threshold transport

AD and PD probes trained as E1, all three source Normal clusters; AD matched C1
is the primary AD arm, other AD arms are sensitivity analyses. Within each IXI
site, freeze a deterministic 50/50 subject split into target-negative calibration
pool and final evaluation, seed 7101. No person belongs to both pools. Entire
sites are external to training; site-specific threshold calibration is explicitly
target adaptation. Publish subject roles in `ixi_roles.csv`. Draw identical
subject IDs across encoders, tasks and source models using seed/site/k/draw.

Six strategies mirror P2: fixed .5; source BA; source 90%-specificity quantile;
source temperature with .5; target empirical 90% quantile; target one-sided
95%-confidence binomial tolerance quantile. k∈{0,4,8,16,32} subjects ×20 draws.
k=0 falls back to source-specificity90; k exceeding pool size gets explicit
`insufficient_target_subjects`. Finite tolerance bound requires ≥29 controls;
unsupported draws get `bound_unavailable`, not a substituted estimate. Use the
existing P2 nextafter tie rule and ≥threshold positive decision.

Primary: specificity and absolute deviation from .90 on untouched final IXI
subjects, per site; worst-site specificity secondary. Paired primary contrast:
target empirical k=32 versus source-specificity90 in absolute specificity error,
averaged over the 20 paired draws, conditional on the fixed calibration pool.
Bootstrap final subjects once per resample and evaluate both strategies on the
same resampled subjects; draws are not 20 independent cohorts. Calibration-pool
sampling variation is described separately by draw quantiles, not a subject CI.
Holm-adjusted paired sign/randomization tests at the subject level for change in
specificity (4 encoders×2 tasks×3 sites); report absolute-error bootstrap contrast
CI separately. Primary AD conclusions restrict to C1; PD cluster results are
descriptive stress analyses. Hypothesis: source specificity changes across IXI
sites and small target-negative samples can restore its operating point.
Falsification: source specificity already transports; calibration fails to reduce
absolute error on held-out controls; or improvement requires impractical k.

IXI has no AD/PD positives: **AUROC, sensitivity, BA, disease detection benefit and
clinical utility are not identifiable in E3** and must be missing, not zero.
Raising a threshold can improve specificity while destroying sensitivity.
Positive temperature cannot change decisions at .5 and is a calibration control.

## Uncertainty, multiplicity and execution

Report percentile 95% CIs from 2,000 stratified subject bootstrap resamples for
E1/E2 (one subject score each); ordinary subject bootstrap for E3 specificity.
These are conditional on the fitted model and calibration draw, not training
uncertainty. Source intervals are by proxy group. Never pool folds, seeds,
encoders, slices or calibration draws as independent people. Smoke uses 200
bootstrap/permutation draws for speed; it is not confirmatory inference.
Two-sided tests, α=.05; Holm families as above. E1 is estimation-first, with
no selected post-hoc winner or uncorrected significance catalogue. Full
inferential contrasts belong in the analysis stage before manuscript claims.
Conditional permutation/randomization test families use fold 0 / first seed /
reference source cluster C1 only; all other fits contribute descriptive
uncertainty/ranges. This avoids choosing a favourable training repetition.

GPU embeddings cached once per encoder/domain, using P2's models, loader,
precision, seed and hash-verified cache convention. CPU probes parallelize
encoder/task/fold/seed/source-cluster and E2 direction/fold/seed, one BLAS thread
per fit; scores are reused across target draws. Every run records commit,
config hash, artifacts, package versions, device and execution scope. Rerun a
probe set in a fresh directory and compare table bytes. Vega uses a separately
built external SIF; CPU probes run in CPU partition, four single-GPU extractions
pack into one node. Confirmed settings and the CPU/GPU/CPU chain are in RUN_P2_external.md.
