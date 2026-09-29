"""Paths and constants shared by every script."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_FAKE = DATA_DIR / "politifact_fake.csv"
RAW_REAL = DATA_DIR / "politifact_real.csv"
PROCESSED = DATA_DIR / "processed" / "headlines.csv"
WIKI_CACHE = DATA_DIR / "wikipedia_cache.json"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models" / "verdict-model"

SEED = 42
N_FOLDS = 5
TOP_K = 3

FAKE, REAL = 0, 1
LABEL_NAMES = {FAKE: "Fake", REAL: "Real"}
