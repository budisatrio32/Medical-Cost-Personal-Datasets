"""Konfigurasi terpusat untuk project Medical Cost Prediction."""
from pathlib import Path

# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_FILE = RAW_DIR / "insurance.csv"
CLEAN_FILE = PROCESSED_DIR / "insurance_clean.csv"
TRAIN_FILE = PROCESSED_DIR / "train.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"

# Fallback kalau file lokal tidak ada (misalnya di Colab tanpa clone repo)
GITHUB_USER = "budisatrio32"
GITHUB_REPO = "Medical-Cost-Personal-Datasets"
GITHUB_BRANCH = "main"
RAW_URL = (
    f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/"
    f"{GITHUB_BRANCH}/data/raw/insurance.csv"
)
REPO_URL = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}.git"

# ---------------------------------------------------------------------------
# Eksperimen
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
STRATIFY_COL = "smoker"

TARGET = "charges"
LOG_TARGET = "log_charges"

# ---------------------------------------------------------------------------
# Aturan validasi
# ---------------------------------------------------------------------------
EXPECTED_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
CATEGORICAL_VALUES = {
    "sex": {"female", "male"},
    "smoker": {"yes", "no"},
    "region": {"northeast", "northwest", "southeast", "southwest"},
}
NUMERIC_RANGES = {
    "age": (18, 100),
    "bmi": (10, 70),
    "children": (0, 10),
    "charges": (0, float("inf")),
}

# ---------------------------------------------------------------------------
# Set fitur untuk eksperimen
# ---------------------------------------------------------------------------
NUMERIC_BASE = ["age", "bmi", "children"]
CATEGORICAL_BASE = ["sex", "smoker", "region"]

FEATURE_SETS = {
    # Versi paper: fitur asli tanpa rekayasa
    "paper": {
        "numeric": ["age", "bmi", "children"],
        "categorical": ["sex", "smoker", "region"],
    },
    # Fitur asli (sama dengan paper, dipisah untuk baseline project)
    "original": {
        "numeric": NUMERIC_BASE,
        "categorical": CATEGORICAL_BASE,
    },
    # Fitur asli + fitur inovasi
    "innovation": {
        "numeric": NUMERIC_BASE + ["obese", "smoker_obese"],
        "categorical": CATEGORICAL_BASE + ["bmi_category", "age_group"],
    },
}
