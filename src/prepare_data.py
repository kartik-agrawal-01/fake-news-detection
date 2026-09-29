"""Build data/processed/headlines.csv from the two raw PolitiFact CSVs.

Every headline gets:
- ``group``: an id shared by exact-duplicate headlines (73 of the 1,056 rows are duplicates), so that
  duplicates never end up on both sides of a train/test split;
- ``fold``: its test fold (0-4) under stratified, grouped 5-fold cross-validation with a fixed seed;
- ``orig_split``: train / val / test / dropped, reproducing the original 80/10/10 split made by the first
  version of this project (Main_dataset_cleaning_and_loading.py), so the old numbers stay comparable.

Usage:
    python -m src.prepare_data
"""
import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from . import config
from .text import normalize_headline


def load_raw(fake_path=config.RAW_FAKE, real_path=config.RAW_REAL) -> pd.DataFrame:
    """Raw headlines in the original order (fake first, then real) with labels 0 = fake, 1 = real."""
    fake = pd.read_csv(fake_path)[["id", "title"]].assign(label=config.FAKE)
    real = pd.read_csv(real_path)[["id", "title"]].assign(label=config.REAL)
    df = pd.concat([fake, real], ignore_index=True).rename(columns={"title": "headline"})
    df["headline"] = df["headline"].fillna("").astype(str)
    return df


def load_nltk_stopwords() -> set[str]:
    """The NLTK English stopword list that the original preprocessing used."""
    import nltk

    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))


def empty_after_original_cleaning(text: str, stop_words: set[str]) -> bool:
    """True if the original preprocessing (letters only, no stopwords, words > 2 chars) left nothing.

    The original script dropped these rows before splitting, so we need the same rule to reproduce its split.
    Lemmatisation never empties a word, so it can be skipped here.
    """
    words = re.sub(r"[^a-zA-Z\s]", "", str(text).lower()).split()
    return not any(w not in stop_words and len(w) > 2 for w in words)


def original_split(df: pd.DataFrame, stop_words: set[str], seed: int = config.SEED) -> pd.Series:
    """Re-create the original train/val/test assignment (same shuffle, same seeds, same stratification)."""
    shuffled = df.sample(frac=1, random_state=seed)
    keep = ~shuffled["headline"].apply(empty_after_original_cleaning, stop_words=stop_words)
    kept = shuffled[keep]
    train_val, test = train_test_split(kept, test_size=0.1, random_state=seed, stratify=kept["label"])
    train, val = train_test_split(train_val, test_size=0.111, random_state=seed, stratify=train_val["label"])
    split = pd.Series("dropped", index=df.index)
    split[train.index] = "train"
    split[val.index] = "val"
    split[test.index] = "test"
    return split


def assign_groups(df: pd.DataFrame) -> pd.Series:
    """Integer group id per distinct normalised headline."""
    return df["headline"].map(normalize_headline).factorize()[0]


def assign_folds(labels, groups, n_folds: int = config.N_FOLDS, seed: int = config.SEED) -> pd.Series:
    """Stratified, grouped K-fold: returns the test-fold index of every row."""
    labels = pd.Series(labels).reset_index(drop=True)
    folds = pd.Series(-1, index=labels.index)
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, test_idx) in enumerate(cv.split(labels, labels, groups=groups)):
        folds[test_idx] = k
    return folds


def prepare(fake_path=config.RAW_FAKE, real_path=config.RAW_REAL, stop_words=None) -> pd.DataFrame:
    df = load_raw(fake_path, real_path)
    df["group"] = assign_groups(df)
    df["fold"] = assign_folds(df["label"], df["group"]).values
    df["orig_split"] = original_split(df, stop_words if stop_words is not None else load_nltk_stopwords())
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(config.PROCESSED))
    args = parser.parse_args()

    df = prepare()
    sizes = df["orig_split"].value_counts().to_dict()
    # Sanity check against the split sizes reported by the original project.
    assert (sizes["train"], sizes["val"], sizes["test"], sizes["dropped"]) == (840, 106, 106, 4), sizes

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Wrote {len(df)} headlines to {out}")
    print(f"  labels: {df['label'].value_counts().sort_index().to_dict()} (0 = fake, 1 = real)")
    print(f"  duplicate headlines: {df['group'].duplicated().sum()} rows share a group with an earlier row")
    print(f"  CV folds: {df.groupby('fold')['label'].agg(['size', 'mean']).round(3).to_dict('index')}")
    print(f"  original split: {sizes}")


if __name__ == "__main__":
    main()
