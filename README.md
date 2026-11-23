# Control-cohort shortcuts and decision-threshold reliability in deep learning for neurodegenerative MRI

Reproducibility code for the manuscript under review by Marko Simonič, Tadej
Peršak, Jernej Hernavs and Simon Klančnik. Correspondence: marko.simonic@um.si.
The paper text is not distributed. This repository contains only its analysis
code, frozen manifests, environment definitions and portable HPC launchers.

- **Part A / P1:** six fine-tuned backbones, changes to control composition and
  training repairs; 1,350 planned fits (810 core, 540 repair), with prespecified
  paired estimates, bootstrap intervals and separate 18/9 Holm families.
- **Part B / P2 internal:** four frozen encoders, 900 score fits across five
  tasks, fixed folds/seeds and source-control strata; threshold transport,
  calibration budgets, paired conditional intervals and explicit unavailable rows.
- **External E1–E4:** NeuroMRI→NEUROCON/Tao Wu PD; cross-cohort PD training;
  IXI site-specific negative calibration (including score-blind age matching);
  and Zenodo AD discrimination/threshold transport. Separate 8/24/8 families.

## Data and weights

No MRI images, masks, clinical archives, embeddings, fitted checkpoints or
analysis results are redistributed. Obtain the exact public releases yourself.
The committed CSVs contain publisher filenames/de-identified participant IDs,
released age/sex fields, derived proxy groups, folds, roles, hashes and summaries;
they add no recovered identity or private participant information.

| Dataset | Obtain release | Paper use |
| --- | --- | --- |
| NeuroMRI v1 | https://data.mendeley.com/datasets/pkmft4d27w/1 (DOI 10.17632/pkmft4d27w.1; publisher CC BY 4.0) | Parts A/B, 25,963 rendered images; six unresolved cross-label images excluded |
| NEUROCON and Tao Wu | https://fcon_1000.projects.nitrc.org/indi/retro/parkinsons.html (CC BY-NC-SA 4.0) | E1/E2, 43/40 subjects |
| IXI | https://brain-development.org/ixi-dataset/ (CC BY-SA 3.0) | E3, 581 healthy volunteers |
| Zenodo 3935636 v1 | https://zenodo.org/records/3935636 (CC BY 4.0) | E4, 35 AD cases and 19 controls; bipolar records excluded |

Review each publisher's terms. NeuroMRI does not document pooled-source
provenance. The external download URLs, byte counts and SHA-256 values are in
`manifests/external/downloads.csv`; `access.json` records source access terms.
`weights/manifest.yaml` pins pretrained releases and their hashes. Only the
encoders named below are fetched; optional gated/non-paper encoders are not needed.

## Environment

Use Linux, Apptainer, SLURM and A100 GPUs for paper-equivalent production.
Python 3.11/3.12, PyTorch 2.5.1, CUDA 12.4, exact numerical pins in
`env/requirements.lock`; external preprocessing adds
`env/external.requirements.lock` and `env/external.build.lock`.
The OCI digests and installations are in `env/neuromri.def` and
`env/neuromri-external.def`. Build on a site-approved build host:

```bash
bash env/build_sif.sh env/neuromri.sif
apptainer build --fakeroot env/neuromri-external.sif env/neuromri-external.def
sha256sum env/neuromri-external.sif > env/neuromri-external.sif.sha256
apptainer exec env/neuromri-external.sif python -m pip freeze > env/external.pip-freeze.txt
```

Archive both exact SIFs, checksums and freezes outside git. The main build also
writes `env/neuromri.pip-freeze.txt`; each analysis writes its runtime
`pip-freeze.txt`. These are generated environment attestations, not a promise
that independent SIF packaging will be byte-identical.

CPU software tests require no datasets, downloads, GPU or scheduler. Use the
pinned environment; for a separate CPU installation:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r env/requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps nibabel==5.3.2 importlib_resources==6.5.2 xlrd==2.0.1 xxhash==3.5.0
python -m pip install --no-deps --no-build-isolation -e .
CUDA_VISIBLE_DEVICES='' python -m pytest tests -q
```

Tests cover frozen group isolation, statistical boundaries, paired bootstrap
conventions, synthetic E1–E4 integration, deterministic repeats, archive safety,
cache identity, entry-point importability and rejection of altered attestation
inputs. The original-history integration test is excluded because the private
history is deliberately not distributed; historical unit fixtures are synthetic.

## Reproduce from downloads to numerical tables

Run from the repository root. Use the committed manifests; regeneration would
change the experiment. Download/extract NeuroMRI v1 so that `DATA_ROOT` directly
contains `Normal/`, `AD Mild/`, and the other publisher class directories.
Do not alter the released images or mix another dataset version.

1. Configure the site. Copy `slurm/hpc.local.env.example` to
   `slurm/hpc.local.env`, fill every placeholder, and retain it locally.
   Allocation, partitions, storage paths and optional QOS are site policy.
   The public environment contains no actual account or partition values.
2. Prepare weights and verify the internal data on the login/build host:

```bash
source slurm/hpc.env
source slurm/hpc.local.env
export NEUROMRI_DATA="$DATA_ROOT" NEUROMRI_WEIGHTS="$WEIGHTS_ROOT"
apptainer exec --bind "$REPO:$REPO" --bind "$DATA_ROOT:$DATA_ROOT:ro" --pwd "$REPO" "$SIF"   python -m neuromri.verify_data
apptainer exec --bind "$REPO:$REPO" --pwd "$REPO" "$SIF"   python -m neuromri.fetch_weights --dest "$WEIGHTS_ROOT"   --encoders resnet18 resnet50 convnext_tiny efficientnet_v2_s vit_b_16 swin_t dinov2_s dinov2_b biomedclip
```

3. Submit Part B extraction, wait for success, then submit the CPU score grid.
   Submit Part A once using the full frozen 1,350-row run list. Four-GPU jobs run
   four independent fits, not distributed training. The one-GPU array is an
   alternative to `four` and must not duplicate the same run list.

```bash
bash slurm/submit.sh extract
# Wait for completed, verified embedding caches.
bash slurm/submit.sh cpu
bash slurm/submit.sh four
# Alternative allocation granularity: bash slurm/submit.sh array
```

4. Run the external chain after setting external raw/tool paths in the local
   environment. On the login host, `bash tools/run_external.sh --download`
   verifies/downloads archives, locks inputs and submits CPU preprocessing →
   four independent encoder jobs → CPU E1–E4 inference. Use a fresh result
   directory. See `docs/RUN_P2_external.md` for outputs and checks.
5. After all jobs complete, enter a SLURM CPU allocation, source
   `slurm/runtime.sh`, and invoke the numerical driver in the main SIF:

```bash
source slurm/runtime.sh
"${CONTAINER[@]}" "$SIF" python tools/reproduce_statistics.py   --p1-root results/p1 --p2-root results/p2   --external-root "$EXTERNAL_RUN_DIR" --output results/paper_tables
```

The driver uses the original Part A command list and `p1_stats`, the original
internal P2 aggregation and paired-draw estimator, and the external CSVs already
computed by `p2_external`. It produces numerical tables, not manuscript files.
Inputs are read-only; choose a fresh output path. The full external manifest must
be complete with unchanged input/output hashes. Runtime provenance records this
public checkout's actual commit. To inspect Part A commands before execution:
`python tools/p1_analysis_commands.py --commands results/p1_commands.sh`.

| Paper result | Script/config | Numerical output |
| --- | --- | --- |
| Part A support and descriptive performance; Table 1, Figures 2–3, Tables S1–S7 | `p1_finetune`, `run_row`, `configs/p1.yaml`, `p1_runs.csv`, `aggregate` | per-fit metrics/predictions/selection/history; `partA/summary.csv` |
| Part A 18 core/9 repair estimates and Holm tests | `neuromri.p1_stats`, `docs/P1_ANALYSIS_PLAN.md` | `partA/inference/estimates.csv`, `fit_deltas.csv`, `analysis.json` |
| Internal threshold transport; Figure 4, Tables S8/S20/S21/S28 | `extract_embeddings`, `p2_threshold_transport`, `configs/p2.yaml`, `tools/p2_internal_fill.py` | `partB_internal/summary.csv`, `availability.csv`, `internal_contrasts.csv`; full `results/p2/metrics.csv` |
| E1 PD and E2 cross-cohort PD; Table 2A, Figure 5, Tables S23/S24 | `p2_external`, `configs/p2_external.yaml` | `E1_metrics.csv`, `E1_contrasts.csv`, `E2_metrics.csv`, `E2_tests.csv` |
| E3 IXI and E4 AD; Table 2B, Figure 6, Tables S25/S26 | same external config, `external_demographics` | `E3_metrics.csv`, `E3_contrasts.csv`, `E4_metrics.csv`, `E4_tests.csv` |
| External roles, demographics and shift diagnostics | `external_data`, `external_diagnostics`, `export_external_roles` | subjects/slices, role CSVs, demographic and diagnostic CSVs |
| Provenance and Supplementary Appendices S1–S5 | plans plus three attestation documents in `docs/`; gates in `tools/` | input inventories, gate JSON, complete run manifest; historical caveat below |

The original manuscript tables combine/select/reformat these numerical CSVs.
No paper wording, Word builders or unpublished-study code is part of this release.
Do not pool draws as independent subjects or treat proxy groups as verified people.
Part A reuses Part B evaluation images and is a follow-up, not independent confirmation.

## Expected compute

The paper's Supplementary Table S7 reports 58.60 occupied A100 GPU-hours of
training across the 1,350 fits: ResNet50 9.08, ConvNeXt-T 12.04,
EfficientNetV2-S 8.58, ViT-B/16 14.08, Swin-T 8.10, DINOv2-B/14 6.72.
Peak allocated memory was 1.96–6.57 GB; mean executed epochs ranged from 9.9 to
15.4 (maximum 30; early stopping patience 5). This is summed occupied training
time, excluding embeddings, external preprocessing, queue time and CPU inference.
The latter stages need additional resources; no unsupported total wall-time
estimate is supplied. Four occupied GPUs count four GPU-hours per wall hour.
Internal P2 fits 900 score models; statistical resampling uses 2,000 draws,
calibration uses 20 draws per supported budget. External preprocessing is CPU
work and uses its separate toolchain; storage must accommodate original archives,
extracted volumes, masks, processed slices and fresh feature caches.

## Provenance

Results were produced at development commits `ada9d38` (P1/P2 internal reporting
reference; P1 training itself at its P1-equivalent parent `469ee8f`) and `808ca5e`
(external) of the private development repository. This public snapshot is derived
from development commit `75eff81d1904a6605109414e2896d7d850e6df77` (`origin/main`).
A git diff across **all retained source paths** verifies no change to training,
scoring, calibration, endpoint formulas, inference, seeds, config values or frozen
manifests between those reporting commits and that baseline. The computational
module difference from `ada9d38` is the later provenance-field/file-based git
lookup and fingerprint addition in `common.py`; it is absent from the diff
against `808ca5e`. Both already contain the documented joblib package-entry fix.

The complete included-path diff also found later SLURM per-step-memory fixes and
provenance/fanout regression tests. The release adds portable paths, redacted historical
gates, validator extraction and the gated local-CPU provenance option. These
are documented exhaustively in `docs/PUBLIC_SNAPSHOT.md`; it would be inaccurate
to claim that the entire release differs only by the two computational fixes.
Historical gates require the private object database and original attestation
exports. They fail closed without that evidence; this fresh history does not
pretend to reproduce original scheduler/reflog records. Fresh runs instead bind
their own public commit, configs, data, weights, SIF and output hashes.

No licence file is supplied pending the PI's choice. Dataset/weight terms remain
separate. The corresponding author's citation metadata is in `CITATION.cff`.
