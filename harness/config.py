"""Central configuration for the black-box object-detection verification harness.

Everything that is specific to a model, a dataset, a device or an acceptance
threshold lives here, so the verification logic stays generic. The defaults run
out-of-the-box on the bundled Ultralytics ``yolo11n-obb`` model and a 300-image
DOTAv1 validation subset; re-point them to verify a different detector.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # the code/ directory
MODELS = ROOT / "models"
DATA = ROOT / "data"
RESULTS = ROOT / "results"

MODEL_PATH = MODELS / "yolo11n-obb.pt"  # model under test (auto-downloaded if absent)
VAL_IMAGES = DATA / "images" / "val"
VAL_LABELS = DATA / "labels" / "val"
DATASET_YAML = RESULTS / "dataset.yaml"  # generated at runtime with portable absolute paths

# Fixed inference configuration (the verification baseline).
IMGSZ = 1024
CONF = 0.25
IOU = 0.7
MAX_DET = 300
DEVICE = 0  # CUDA device index, or "cpu"
SEED = 42

# Class names of the model under test (index -> name). The bundled model is
# DOTAv1 (15 oriented-bounding-box classes); replace this map for another model.
CLASS_NAMES = {
    0: "plane",
    1: "ship",
    2: "storage tank",
    3: "baseball diamond",
    4: "tennis court",
    5: "basketball court",
    6: "ground track field",
    7: "harbor",
    8: "bridge",
    9: "large vehicle",
    10: "small vehicle",
    11: "helicopter",
    12: "roundabout",
    13: "soccer ball field",
    14: "swimming pool",
}
NUM_CLASSES = len(CLASS_NAMES)

# Optional class ids the application cares about, used for coverage/accuracy
# reporting. Empty means "report against every class the model has".
CLASSES_OF_INTEREST: list[int] = []

# ---- acceptance thresholds (edit to match your SRD; (TBC) = confirm with the authority) ----
MIN_OBJECT_PX = 20  # minimum detectable object size (px) for ODD coverage
STD_IMG = 1024  # standardised pipeline input size (px)
INFER_BUDGET_MS = 50.0  # per-frame inference budget
E2E_BUDGET_MS = 100.0  # end-to-end budget
MAX_LABEL_ERROR_PCT = 1.0  # max acceptable label error rate (TC-DATA-04)
DRIFT_ALPHA = 0.05  # KS p-value threshold for drift detection (TC-DATA-05)
ROB_MIN_RETENTION = 0.70  # min mean detection retention (TC-ROB-01 sev1 / TC-ROB-02)

# Sample sizes for the model-dependent sweeps (kept modest for a 4 GB GPU).
ROB_IMAGES = 40
OOD_IMAGES = 100
TIM_IMAGES = 150
FUN_IMAGES = 50
