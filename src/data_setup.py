"""Setup dataset: load -> validasi -> cleaning -> fitur inovasi -> split -> simpan.

Jalankan dari root project:
    python src/data_setup.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_raw(path=cfg.RAW_FILE, url=cfg.RAW_URL):
    """Baca dataset dari file lokal; kalau tidak ada, unduh dari GitHub."""
    path = Path(path)
    if path.exists():
        print(f"[load] Membaca file lokal: {path}")
        return pd.read_csv(path)

    print(f"[load] File lokal tidak ditemukan, mengunduh dari: {url}")
    df = pd.read_csv(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# 2. Validasi
# ---------------------------------------------------------------------------
def validate(df):
    """Cek kolom, nilai kategori, dan rentang nilai numerik."""
    missing = set(cfg.EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Kolom hilang: {missing}")

    for col, allowed in cfg.CATEGORICAL_VALUES.items():
        invalid = set(df[col].dropna().str.strip().str.lower().unique()) - allowed
        if invalid:
            raise ValueError(f"Nilai tidak valid di '{col}': {invalid}")

    for col, (lo, hi) in cfg.NUMERIC_RANGES.items():
        out = df[(df[col] < lo) | (df[col] > hi)]
        if len(out):
            raise ValueError(f"{len(out)} nilai '{col}' di luar rentang [{lo}, {hi}]")

    print(f"[validate] OK - {df.shape[0]} baris, {df.shape[1]} kolom")
    print(f"[validate] Missing value: {int(df.isna().sum().sum())}")
    print(f"[validate] Baris duplikat: {int(df.duplicated().sum())}")


# ---------------------------------------------------------------------------
# 3. Cleaning
# ---------------------------------------------------------------------------
def clean(df):
    """Rapikan string, hapus missing & duplikat, set tipe data."""
    df = df.copy()
    for col in cfg.CATEGORICAL_VALUES:
        df[col] = df[col].str.strip().str.lower()

    before = len(df)
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    print(f"[clean] Menghapus {before - len(df)} baris (missing/duplikat) -> {len(df)} baris")

    df["age"] = df["age"].astype(int)
    df["children"] = df["children"].astype(int)
    return df


# ---------------------------------------------------------------------------
# 4. Fitur inovasi
# ---------------------------------------------------------------------------
def add_features(df):
    """Tambah fitur obese, smoker_obese, bmi_category, age_group, log_charges."""
    df = df.copy()
    df["obese"] = (df["bmi"] >= 30).astype(int)
    df["smoker_obese"] = ((df["smoker"] == "yes") & (df["obese"] == 1)).astype(int)

    # Kategori BMI menurut WHO
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25, 30, np.inf],
        labels=["underweight", "normal", "overweight", "obese"],
        right=False,
    ).astype(str)

    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 29, 39, 49, 59, np.inf],
        labels=["18-29", "30-39", "40-49", "50-59", "60+"],
    ).astype(str)

    df[cfg.LOG_TARGET] = np.log1p(df[cfg.TARGET])

    print(
        f"[features] Skewness charges: {df[cfg.TARGET].skew():.2f} -> "
        f"log_charges: {df[cfg.LOG_TARGET].skew():.2f}"
    )
    return df


# ---------------------------------------------------------------------------
# 5. Split & simpan
# ---------------------------------------------------------------------------
def split(df):
    """Split 80:20 distratifikasi berdasarkan status perokok."""
    train, test = train_test_split(
        df,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
        stratify=df[cfg.STRATIFY_COL],
    )
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    print(f"[split] Train: {len(train)} | Test: {len(test)}")
    print(
        f"[split] Proporsi perokok - train: {(train['smoker'] == 'yes').mean():.3f}, "
        f"test: {(test['smoker'] == 'yes').mean():.3f}"
    )
    return train, test


def save(df, train, test):
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.CLEAN_FILE, index=False)
    train.to_csv(cfg.TRAIN_FILE, index=False)
    test.to_csv(cfg.TEST_FILE, index=False)
    print(f"[save] Tersimpan di {cfg.PROCESSED_DIR}")


# ---------------------------------------------------------------------------
# Helper untuk tahap modeling
# ---------------------------------------------------------------------------
def load_processed():
    """Baca train & test hasil setup. Jalankan run_setup() dulu kalau belum ada."""
    if not (cfg.TRAIN_FILE.exists() and cfg.TEST_FILE.exists()):
        run_setup()
    return pd.read_csv(cfg.TRAIN_FILE), pd.read_csv(cfg.TEST_FILE)


def get_xy(df, feature_set="original", log_target=False):
    """Ambil X (fitur sesuai feature_set) dan y (charges / log_charges)."""
    fs = cfg.FEATURE_SETS[feature_set]
    X = df[fs["numeric"] + fs["categorical"]].copy()
    y = df[cfg.LOG_TARGET if log_target else cfg.TARGET].copy()
    return X, y


def run_setup():
    df = load_raw()
    validate(df)
    df = clean(df)
    df = add_features(df)
    train, test = split(df)
    save(df, train, test)
    return df, train, test


if __name__ == "__main__":
    run_setup()
