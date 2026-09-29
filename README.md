# Fake News Detection with Retrieval-Augmented Generation

A fact-checking prototype. It takes a news headline, retrieves supporting evidence from a knowledge base of verified real headlines and from live Wikipedia, and returns a **Real / Fake** verdict with a short explanation.

Course project at NMIMS. I built the full pipeline: data preparation, retrieval, generation, the Streamlit app and a fine-tuning experiment.

## What it does

- **Streamlit web app** (`app.py`): paste a headline and get a verdict, an explanation, a 0–10 relevance score and the evidence snippets it used.
- **Command-line version** (`interactive_rag.py`): the same pipeline in a terminal loop.
- **Live Wikipedia retrieval:** each query also fetches a 2-sentence Wikipedia summary. This covers recent or global events that aren't in the static knowledge base.

## How it works

```
headline ─► preprocess ─► BM25 over real-news KB (top 3) ─┐
                     └──► Wikipedia summary (live)  ─────┼─► relevance score ─► prompt ─► GPT-2 ─► verdict parser ─► rule-based overrides ─► Real / Fake
```

1. **Data** (`Main_dataset_cleaning_and_loading.py`)
   - Takes the PolitiFact headlines from FakeNewsNet: 432 fake and 624 real.
   - Lower-cases them, strips punctuation, removes stopwords and lemmatises.
   - Splits them 80/10/10 (stratified): 840 train, 106 validation, 106 test.
2. **Retrieval** (`retrieval_module.py`)
   - Builds a BM25 index (`rank_bm25`) over the 495 *real* headlines in the training split, and returns the top 3 matches for a query.
   - If the `wikipedia` package is installed, it also queries Wikipedia live. It includes a small parser for election-style claims and a fallback for disambiguation pages.
   - The Wikipedia result gets a fixed relevance score of 5.0.
3. **Generation** (`generation_module.py`)
   - The evidence and the headline go into a fact-checker prompt for **GPT-2** (base model).
   - `bad_words_ids` and a repetition penalty stop it from echoing the prompt.
   - The verdict is read from the generated text with a regex, falling back to counting keywords.
4. **Decision rules** (`app.py`)
   - The average BM25 score is turned into a 0–10 relevance score, plus a boost when Wikipedia evidence is present.
   - High relevance with Wikipedia present forces **Real**.
   - Low relevance (average BM25 below 3.0) forces **Fake**.
   - A generated "Real" that just echoes the query is flipped to **Fake**.
5. **Fine-tuning experiment** (`fine_tune_model.py`)
   - Fine-tunes DistilGPT-2 on training headlines formatted as fact-checker prompts that end in the correct verdict.
   - The app doesn't use this model; it uses base GPT-2, as the code comment says: "base for stability".
6. **Semantic retriever** (`build_semantic_retriever.py`)
   - An optional alternative: a FAISS index over `all-MiniLM-L6-v2` sentence embeddings.
   - The app doesn't use it.

`Frontend.html` is a static HTML mock-up of the interface, made before the Streamlit app.

## Results

**Accuracy on the PolitiFact test split: 56.6% (60 of 106 headlines).** That's below the 59.4% you'd get by always answering "Real". In its current form, the system doesn't beat a trivial baseline.

| Setting | Accuracy | Macro F1 | Fake precision / recall | Real precision / recall | Confusion matrix `[[TN, FP], [FN, TP]]` |
|---|---|---|---|---|---|
| **App pipeline: BM25 + live Wikipedia** | **56.6%** | 0.565 | 0.475 / 0.651 | 0.681 / 0.508 | `[[28, 15], [31, 32]]` |
| BM25 only (Wikipedia off) | 50.9% | 0.467 | 0.452 / 0.977 | 0.923 / 0.190 | `[[42, 1], [51, 12]]` |
| Majority class (always "Real") | 59.4% | n/a | n/a | n/a | n/a |

### How it was measured
- **Script:** `eval_rag.py`, run on 29 September 2026. It sends every headline in the held-out test split (106 headlines: 63 real, 43 fake) through the same pipeline as `app.py`: BM25 top-3, live Wikipedia, the same prompt, GPT-2, and the same override rules.
- **Queries are the original raw headlines** (what a user would type). The script recovers them by re-running the original split with the same seeds, and asserts that the result matches `test_processed.csv` exactly.
- **The test headlines never appear in the retrieval knowledge base**, which is built only from the training split.
- **Seed:** 42. Generation samples, and Wikipedia is live, so reruns can differ slightly.
- **Wikipedia coverage:** Wikipedia returned a summary for 41 of the 106 headlines. For those, the app's override rule (high relevance plus Wikipedia present → "Real") drives most predictions toward "Real".
- **Outputs:** per-headline predictions and metrics are in [`eval_results/`](eval_results/).
- **Environment:** Python 3.10, torch 2.8.0 (CPU), transformers 4.57.1, wikipedia 1.4.0. Newer torch and transformers than the pins in `requirements.txt`. A full run takes about 13 minutes on a laptop CPU.

### What the numbers show
- **Without Wikipedia, the system labels almost everything "Fake"** (93 of 106). Most real test headlines score low on BM25 against a knowledge base of only 495 headlines, so the "low relevance → Fake" rule fires.
- **Wikipedia pushes predictions toward "Real".** That raises accuracy but not above the majority baseline.
- **GPT-2 has little influence on the verdict.** It isn't instruction-tuned, and the retrieval-score rules override most of its outputs.

The first evaluation run also exposed a crash in the app. When a Wikipedia lookup hit a disambiguation page, the fallback lookup could hit a second one, and nothing caught that error. `retrieval_module.py` now catches it and falls back to the "no additional fact" placeholder.

The DistilGPT-2 fine-tuning log records a best **validation loss** of 0.9215. That's a language-modelling loss, not an accuracy.

## How to run

Requires Python 3.10.

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Then:

```bash
streamlit run app.py
```

On first run:
- The BM25 index (`bm25_retriever.pkl`) is built from `train_processed.csv`.
- GPT-2 is downloaded from Hugging Face.
- NLTK data is downloaded.

Other entry points:

| Command | What it does |
|---|---|
| `python interactive_rag.py` | Command-line version |
| `python Main_dataset_cleaning_and_loading.py` | Rebuild the train/val/test splits from the raw PolitiFact CSVs |
| `python eval_rag.py` | Evaluate on the 106-headline test split (`--no-wiki` for BM25 only, `--limit N` for a quick run). Writes `eval_results/` |
| `python fine_tune_model.py` | Fine-tune DistilGPT-2 (the model isn't committed: it's ~313 MB) |
| `python build_semantic_retriever.py` | Build the optional FAISS index |

`recreate_venv_py310.bat` is a Windows helper that rebuilds the virtual environment with CPU-only PyTorch.

## Data

- `politifact_fake.csv` and `politifact_real.csv` come from **FakeNewsNet** ([KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)): Shu et al., *FakeNewsNet: A Data Repository with News Content, Social Context and Spatiotemporal Information for Studying Fake News on Social Media*, 2018.
- `train_processed.csv`, `val_processed.csv` and `test_processed.csv` are the preprocessed splits made by `Main_dataset_cleaning_and_loading.py`.
- These datasets were explored but aren't used by the pipeline, so they're not committed:
  - The FakeNewsNet GossipCop files
  - The Kaggle [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (`Fake.csv` / `True.csv`, >50 MB each)

## Limitations
- The knowledge base is small: 495 PolitiFact headlines, mostly US political claims. Anything outside it relies on Wikipedia.
- GPT-2 isn't instruction-tuned, so the final verdict leans heavily on the retrieval score and the rule-based overrides.
- Measured accuracy (56.6%) is below the majority-class baseline (59.4%). Better next steps would be a larger knowledge base and a fine-tuned classifier or an instruction-tuned model for the verdict.
- Live Wikipedia lookups make results depend on network access and on current Wikipedia content.

## Tech stack
Python · Streamlit · Hugging Face Transformers (GPT-2, DistilGPT-2) · PyTorch · rank-bm25 · NLTK · wikipedia · scikit-learn · pandas · (optional) sentence-transformers + FAISS
