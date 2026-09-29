"""
Evaluate the RAG fact-checker on the PolitiFact test split.

Each test headline goes through the same pipeline as app.py:
BM25 top-3 (+ live Wikipedia) -> fact-checker prompt -> GPT-2 -> verdict parser -> app.py override rules.

The test split (test_processed.csv) stores preprocessed text (lower-cased, lemmatised, stopwords removed).
Users type raw headlines into the app, so by default we recover the raw headline for every test row by
re-running the split from Main_dataset_cleaning_and_loading.py with the same random seeds.

Usage:
    python eval_rag.py                 # full test split (106 headlines), Wikipedia on
    python eval_rag.py --no-wiki       # BM25-only ablation
    python eval_rag.py --processed     # use the preprocessed text as the query instead of raw headlines
    python eval_rag.py --limit 20      # quick smoke test
"""
import argparse
import json
import os
import random
import re
import time

import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import retrieval_module
from retrieval_module import retrieve_relevant_docs
from generation_module import generate_verdict

SEED = 42


def recover_raw_test_headlines():
    """Rebuild the exact split from Main_dataset_cleaning_and_loading.py and return the raw test headlines."""
    fake_df = pd.read_csv("politifact_fake.csv")
    real_df = pd.read_csv("politifact_real.csv")
    fake_df["statement"] = fake_df["title"]
    real_df["statement"] = real_df["title"]
    fake_df["label"] = 0
    real_df["label"] = 1
    combined_df = pd.concat([fake_df[["statement", "label"]], real_df[["statement", "label"]]], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
        return " ".join(tokens)

    combined_df["clean_text"] = combined_df["statement"].apply(preprocess_text)
    combined_df = combined_df[combined_df["clean_text"].str.len() > 0].reset_index(drop=True)

    _, test_part = train_test_split(combined_df, test_size=0.1, random_state=42, stratify=combined_df["label"])

    # Sanity check: must reproduce test_processed.csv exactly
    saved = pd.read_csv("test_processed.csv")
    assert list(test_part["clean_text"]) == list(saved["text"]), "Recovered split does not match test_processed.csv"
    assert list(test_part["label"]) == list(saved["label"])
    return pd.DataFrame({"query": test_part["statement"].values, "clean_text": test_part["clean_text"].values,
                         "label": test_part["label"].values})


def predict(news_input):
    """Mirror of the decision logic in app.py (retrieval -> prompt -> generation -> override rules)."""
    docs, scores = retrieve_relevant_docs(news_input, k=3)
    if not docs:
        return "Fake", 0.0, False

    avg_score = sum(scores) / len(scores) if scores else 0
    facts_text = "\n".join(docs)

    normalized_score = min(10.0, max(0.0, avg_score / 1.5))
    wikipedia_present = len(scores) > 3 and scores[-1] == 5.0
    if wikipedia_present:
        normalized_score += 3.5
        normalized_score = min(10.0, normalized_score)

    prompt = f"""You are a fact-checker. Use the real facts to evaluate the news.

Real facts: {facts_text}

News: "{news_input}"

Verdict: """

    verdict, explanation = generate_verdict(prompt, score=normalized_score)

    if normalized_score >= 6.0 and wikipedia_present:
        verdict = "Real"
    elif avg_score < 3.0:
        verdict = "Fake"
    elif avg_score < 6.0:
        raw_lower = explanation.lower()
        query_words = news_input.lower().split()[:3]
        echo_detected = any(word in raw_lower for word in query_words)
        if echo_detected and verdict == "Real":
            verdict = "Fake"

    return verdict, normalized_score, wikipedia_present


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-wiki", action="store_true", help="disable live Wikipedia retrieval (BM25 only)")
    parser.add_argument("--processed", action="store_true", help="query with preprocessed text instead of raw headlines")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N test rows")
    parser.add_argument("--out", default="eval_results", help="output folder for predictions and metrics")
    args = parser.parse_args()

    if args.no_wiki:
        retrieval_module.WIKI_ENABLED = False

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    test_df = recover_raw_test_headlines()
    if args.limit:
        test_df = test_df.head(args.limit)
    query_col = "clean_text" if args.processed else "query"

    rows = []
    start = time.time()
    for i, row in enumerate(test_df.itertuples(index=False), 1):
        verdict, score, wiki = predict(getattr(row, query_col))
        pred = 1 if verdict == "Real" else 0
        rows.append({"query": getattr(row, query_col), "label": row.label, "pred": pred,
                     "relevance_score": round(score, 3), "wikipedia_used": wiki})
        print(f"[{i}/{len(test_df)}] label={row.label} pred={pred} score={score:.2f} wiki={wiki}", flush=True)
    elapsed = time.time() - start

    res = pd.DataFrame(rows)
    y, p = res["label"], res["pred"]
    acc = accuracy_score(y, p)
    report = classification_report(y, p, target_names=["Fake", "Real"], output_dict=True, zero_division=0)
    cm = confusion_matrix(y, p, labels=[0, 1]).tolist()
    majority = max(y.mean(), 1 - y.mean())

    tag = ("bm25_only" if args.no_wiki else "bm25_wiki") + ("_processed" if args.processed else "_raw")
    os.makedirs(args.out, exist_ok=True)
    res.to_csv(os.path.join(args.out, f"predictions_{tag}.csv"), index=False)
    metrics = {
        "setting": tag,
        "n": len(res),
        "accuracy": round(acc, 4),
        "majority_class_baseline": round(majority, 4),
        "fake_precision": round(report["Fake"]["precision"], 4),
        "fake_recall": round(report["Fake"]["recall"], 4),
        "fake_f1": round(report["Fake"]["f1-score"], 4),
        "real_precision": round(report["Real"]["precision"], 4),
        "real_recall": round(report["Real"]["recall"], 4),
        "real_f1": round(report["Real"]["f1-score"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "confusion_matrix_[[TN_fake,FP],[FN,TP_real]]": cm,
        "wikipedia_used_count": int(res["wikipedia_used"].sum()),
        "seconds": round(elapsed, 1),
        "date": time.strftime("%Y-%m-%d"),
    }
    with open(os.path.join(args.out, f"metrics_{tag}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + classification_report(y, p, target_names=["Fake", "Real"], zero_division=0))
    print(f"Accuracy: {acc:.2%}  (majority-class baseline: {majority:.2%}, n={len(res)})")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
