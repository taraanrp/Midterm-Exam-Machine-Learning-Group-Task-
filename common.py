"""Fungsi bersama untuk ketiga tugas UTS Machine Learning.

Semua jalur dihitung relatif terhadap root repo, sehingga skrip bisa dijalankan
dari mana saja. Data mentah (*.csv.gz) diletakkan di folder data/ (tidak di-commit).
"""
import json
import os
import random
import time
from pathlib import Path

import numpy as np

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("DATA_DIR", ROOT / "data"))
HASIL = ROOT / "hasil"
HASIL.mkdir(exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def pinball(y, p, q):
    """Pinball loss rata-rata untuk kuantil q."""
    d = np.asarray(y) - np.asarray(p)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def save_json(obj, name):
    with open(HASIL / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)


def load_json(name):
    with open(HASIL / name, encoding="utf-8") as f:
        return json.load(f)


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t = time.time()
        print(f"[{self.label}] mulai", flush=True)
        return self

    def __exit__(self, *a):
        print(f"[{self.label}] selesai {time.time() - self.t:.0f}s", flush=True)
