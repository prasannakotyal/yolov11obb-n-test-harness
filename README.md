# YOLOv11n-OBB Verification Harness

A runnable, end-to-end software harness that executes a set of **black-box** verification
cases against an oriented-bounding-box (OBB) object detector and records the results in
SVCP "Recording of Test Results" format, with every case traced back to the SRD.

It is **model- and dataset-agnostic** - everything specific lives in `harness/config.py`.
Out of the box it runs on the bundled Ultralytics `yolo11n-obb` model and a 300-image
DOTAv1 validation subset, so the reference numbers below reproduce as-is.

Verification is **black-box only**: observable inputs and outputs, no model internals,
weights or gradients. White-box gradient attacks (FGSM/PGD) are intentionally out of scope.

## Requirements
- Python 3.10-3.12 and [`uv`](https://docs.astral.sh/uv/)
- [Git LFS](https://git-lfs.com) - the model and dataset are stored via LFS
- An NVIDIA GPU is optional (the harness falls back to CPU; latency numbers are device-dependent)

## Quickstart
```bash
git lfs install                     # the model + dataset are in Git LFS; install it before cloning
git clone git@github.com:prasannakotyal/yolov11obb-n-test-harness.git
cd yolov11obb-n-test-harness

uv venv .venv
uv pip install -e ".[dev]"          # harness + pytest

uv run python run_harness.py        # run all cases  -> results/
uv run pytest -m "not slow"         # fast unit tests (no model/GPU), ~1 s
uv run pytest                       # also the slow end-to-end case (loads the model)
```
If you cloned before installing Git LFS, run `git lfs pull` to fetch the real files. For GPU
inference, install the torch build matching your CUDA (torch is otherwise pulled in by ultralytics).

## Layout
```
.
├── harness/
│   ├── config.py         # all model/dataset/device/threshold settings (the only file to edit)
│   ├── traceability.py   # test case -> SRD requirement map (single source of truth)
│   ├── metrics.py        # pure verification maths (AUROC, FPR95, drift, retention, schema)
│   ├── corruptions.py    # 15 corruptions x 5 severities + geometric transforms (pure image ops)
│   ├── common.py         # model load/predict, image IO, result recording
│   ├── data_tests.py     # TC-DATA-01..05   (data-management verification)
│   ├── robustness.py     # TC-ROB-01..02    (corruption + geometric robustness)
│   ├── model_tests.py    # TC-PERF-01, TC-TIM-01, TC-FUN (accuracy, latency, schema)
│   └── ood.py            # TC-OOD-01        (out-of-distribution rejection)
├── tests/                # pytest suite (unit tests + one marked-slow integration test)
├── models/               # model under test (yolo11n-obb.pt; auto-downloaded if absent)
├── data/                 # images/val + labels/val (DOTA-format OBB labels)
├── results/              # svcp_results.{csv,md}, summary.json, <TC-ID>/ evidence
└── run_harness.py        # orchestrator
```

## What it implements

**Data-management verification**
- `TC-DATA-01` dataset integrity + class distribution (pairing, MD5 duplicates, malformed/out-of-range labels, imbalance)
- `TC-DATA-02` ODD coverage + desiderata (per-class coverage, small-object fraction, image-size range)
- `TC-DATA-03` train/val/test separation (MD5 exact + 16x16 average-hash near-duplicate detection)
- `TC-DATA-04` label-quality review (format + polygon-geometry validity over every object)
- `TC-DATA-05` data-drift monitoring (KS + Jensen-Shannon on brightness; control vs induced drift)

**Black-box model validation**
- `TC-ROB-01` corruption robustness: 15 corruptions x 5 severities; detection-retention curve
- `TC-ROB-02` geometric robustness (metamorphic): flip / rotate / brightness / zoom-out
- `TC-OOD-01` OOD rejection: uniform-noise + block-scrambled inputs; AUROC + FPR@95%TPR
- `TC-PERF-01` per-class accuracy via `model.val()` (mAP50, mAP50-95, precision, recall)
- `TC-TIM-01` inference + end-to-end latency (mean + p95)
- `TC-FUN` detect/classify + output-schema validity

Interface and failure-injection cases need a hardware test rig and are out of scope here.

## How results map to the SRD
Each case maps to one or more SRD requirements via `harness/traceability.py`, the single source
of truth. The map uses neutral group codes (`FR` functional, `PR` performance, `TR` timing,
`IR` interface, `OR` operational, `CR` constraints, `FD` failure-detection, `DF` data/format,
`AI` AI/ML, `SR` safety, `DR` dataset). Edit that one file to match your SRD; the IDs then appear
in the "Traceability (SRD)" column of `results/svcp_results.{csv,md}` and in each `result.json`.

The pytest suite verifies the methodology against known-answer inputs (e.g. AUROC of a perfectly
separable set is 1.0, a unit-square polygon area is 1.0, KS flags an induced shift but not a
control). The slow test runs one real model case and auto-skips when no model/GPU is present.

## Configuring for a different model / dataset / thresholds
Edit `harness/config.py`:
- `MODEL_PATH`, `VAL_IMAGES`, `VAL_LABELS`, `CLASS_NAMES`, `DEVICE`, inference parameters
- `CLASSES_OF_INTEREST` to restrict coverage/accuracy reporting to a target subset (empty = all)
- the acceptance thresholds block (`INFER_BUDGET_MS`, `ROB_MIN_RETENTION`, `OOD_MIN_AUROC`, ...)
- sample sizes for the model-dependent sweeps (`ROB_IMAGES`, `OOD_IMAGES`, `TIM_IMAGES`, `FUN_IMAGES`)

The Ultralytics data YAML is generated at runtime from config (`results/dataset.yaml`), so the
repo carries no machine-specific path.

## Reference run
`yolo11n-obb` on the bundled 300-image DOTAv1 val subset (NVIDIA RTX 3050 Laptop GPU):

| Case | Result | Key measured value(s) |
|------|--------|-----------------------|
| TC-DATA-01 | PASS | 18,451 instances, 15 classes, imbalance 82.4, 0 duplicates |
| TC-DATA-02 | PASS | 15/15 classes >=5 instances; 24.9% objects <20 px |
| TC-DATA-03 | PASS | 0 exact / 0 near duplicates within val |
| TC-DATA-04 | PASS | label error rate 0.0% over 18,451 objects |
| TC-DATA-05 | PASS | induced drift KS p=3e-47 (detected); control p=0.36 (not flagged) |
| TC-ROB-01 | PASS | retention sev1 0.95 -> sev3 0.75 -> sev5 0.48 (graceful) |
| TC-ROB-02 | PASS | mean retention 0.98 over 8 transforms |
| TC-FUN | PASS | 98% images with detections; schema valid 49/50 |
| TC-TIM-01 | PASS | inference 16.2 ms (p95 67.1); end-to-end 22.9 ms |
| TC-PERF-01 | MEASURED | mAP50 0.618, mAP50-95 0.467, P 0.809, R 0.581 (per-class in accuracy.json) |
| TC-OOD-01 | MEASURED | AUROC 0.776, FPR@95TPR 1.0, OOD confident-detection rate 0.49 |

`MEASURED` means the mechanism is proven and values are reported; the pass/fail threshold is set
against the SRD acceptance values once confirmed.

## Assumptions and limitations
- The bundled DOTAv1 `yolo11n-obb` is a stand-in to make the harness reproducible; re-point
  `config.py` at the model and labelled dataset actually under verification.
- All `(TBC)` thresholds are placeholders to confirm with the certification authority; reported
  numbers are **measured**, not assumed.
- `TC-OOD-01` uses a simple max-confidence black-box score; it already surfaces a real weakness
  (about half of block-scrambled OOD inputs still trigger a confident detection), which motivates
  a dedicated OOD scorer (ODIN/energy).
- Latency depends on the host device (reported in the results header); final acceptance is on the
  deployment target hardware.
- Randomised cases (`TC-ROB`, `TC-OOD`) are seeded (`config.SEED`) for reproducibility but may
  shift slightly across GPU/driver versions.

> The model (`models/`, ~6 MB) and the validation subset (`data/`, ~215 MB) are tracked with
> **Git LFS** so the harness runs out-of-the-box. Install Git LFS before cloning (`git lfs install`);
> if you already cloned without it, run `git lfs pull`.
