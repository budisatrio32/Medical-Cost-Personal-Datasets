# Medical Cost Personal Datasets - Prediksi Biaya Asuransi

Project Praktikum Penambangan Data: prediksi `charges` (biaya medis) dari dataset Medical Cost Personal (1.338 baris).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/budisatrio32/Medical-Cost-Personal-Datasets/blob/main/notebooks/01_data_setup.ipynb)

## Struktur

```
data/
  raw/insurance.csv        # dataset asli
  processed/               # hasil setup (tidak di-commit, dibuat ulang otomatis)
notebooks/
  01_data_setup.ipynb      # setup dataset, bisa di lokal maupun Colab
src/
  config.py                # path, seed, rasio split, aturan validasi, set fitur
  data_setup.py            # load -> validasi -> cleaning -> fitur -> split -> simpan
requirements.txt
```

## Menjalankan

**Google Colab:** klik badge di atas, lalu *Runtime → Run all*. Sel pertama otomatis meng-clone repo ini dan meng-install library.

**Lokal:**
```bash
pip install -r requirements.txt
python src/data_setup.py
```

Output: `data/processed/insurance_clean.csv`, `train.csv` (1.069 baris), `test.csv` (268 baris).

## Pipeline setup

1. **Load**: dari `data/raw/insurance.csv`; kalau tidak ada, diunduh dari repo GitHub ini.
2. **Validasi**: kolom, nilai kategori, rentang numerik.
3. **Cleaning**: rapikan string, hapus missing & duplikat (1 baris duplikat dihapus → 1.337 baris).
4. **Fitur inovasi**: `obese`, `smoker_obese`, `bmi_category` (WHO), `age_group`, `log_charges` (skewness 1,52 → -0,09).
5. **Split** 80:20, stratify `smoker`, `random_state=42`.

## Dipakai di notebook modeling

```python
import sys; sys.path.insert(0, "src")
from data_setup import load_processed, get_xy

train, test = load_processed()          # otomatis menjalankan setup kalau belum ada
X_train, y_train = get_xy(train, feature_set="innovation", log_target=True)
X_test,  y_test  = get_xy(test,  feature_set="innovation", log_target=True)
```

Set fitur: `paper`, `original`, `innovation` (lihat `src/config.py`).
