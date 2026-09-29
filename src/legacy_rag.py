"""The original pipeline: BM25 over real headlines + Wikipedia -> prompt -> GPT-2 -> verdict parser -> override rules.

Ported from the first version of the project (retrieval_module.py, generation_module.py, app.py and eval_rag.py)
with the same preprocessing, prompt, generation settings and rules. Two changes make it evaluable under
cross-validation: Wikipedia results come from the cache, and each headline gets its own generation seed, so a
prediction does not depend on the order in which headlines are processed.

``apply_rules`` also records which rule decided the verdict, which is how the evaluation measures how much GPT-2
actually contributes.
"""
import re

import numpy as np

from . import wiki as wiki_mod
from .verdict import parse_generated_verdict

GPT2_BAD_WORDS = ["Source:", "Fact:", "Relevant Real Facts:", "You are a fact-checker", "If facts match",
                  "Use only these", "Verdict:", "Obama administration", "funding the Islamic State", "The White House",
                  "The truth is", "Example", "Claim:"]

RULE_WIKI_REAL = "wiki_high_relevance_real"
RULE_LOW_FAKE = "low_relevance_fake"
RULE_ECHO_FAKE = "echo_flipped_to_fake"
RULE_NONE = "gpt2_verdict_kept"


def ensure_nltk():
    import nltk

    for resource, name in [("tokenizers/punkt_tab", "punkt_tab"), ("corpora/stopwords", "stopwords"),
                           ("corpora/wordnet", "wordnet")]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(name, quiet=True)


class LegacyPreprocessor:
    """Both NLTK preprocessing steps of the original: one for building the processed dataset, one for BM25."""

    def __init__(self):
        ensure_nltk()
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize

        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.word_tokenize = word_tokenize

    def clean_text(self, text: str) -> str:
        """Main_dataset_cleaning_and_loading.py: the knowledge base stored this form of each headline."""
        text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
        return " ".join(self.lemmatizer.lemmatize(w) for w in self.word_tokenize(text)
                        if w not in self.stop_words and len(w) > 2)

    def bm25_tokens(self, text: str) -> list[str]:
        """retrieval_module.preprocess_text: tokens for BM25 (queries and knowledge-base documents)."""
        return [self.lemmatizer.lemmatize(t, pos="v") for t in self.word_tokenize(str(text).lower())
                if t.isalpha() and t not in self.stop_words]


def apply_rules(verdict: str, explanation: str, headline: str, scores: list[float]):
    """The app's override rules. Returns (verdict, explanation, rule, normalized_score, wikipedia_present)."""
    avg_score = sum(scores) / len(scores) if scores else 0.0
    normalized = min(10.0, max(0.0, avg_score / 1.5))
    wikipedia_present = len(scores) > 3 and scores[-1] == 5.0
    if wikipedia_present:
        normalized = min(10.0, normalized + 3.5)

    if normalized >= 6.0 and wikipedia_present:
        return ("Real", f"Supported by high-relevance facts including Wikipedia (score: {normalized:.2f}).",
                RULE_WIKI_REAL, normalized, wikipedia_present)
    if avg_score < 3.0:
        return ("Fake", explanation or "Low relevance in facts; likely unsubstantiated.",
                RULE_LOW_FAKE, normalized, wikipedia_present)
    if avg_score < 6.0 and verdict == "Real":
        query_words = headline.lower().split()[:3]
        if any(word in explanation.lower() for word in query_words):
            return ("Fake", f"Lacks evidence (echo detected, score: {normalized:.2f}).",
                    RULE_ECHO_FAKE, normalized, wikipedia_present)
    return verdict, explanation, RULE_NONE, normalized, wikipedia_present


def build_prompt(headline: str, docs: list[str]) -> str:
    facts_text = "\n".join(docs)
    return f"""You are a fact-checker. Use the real facts to evaluate the news.

Real facts: {facts_text}

News: "{headline}"

Verdict: """


class LegacyRAG:
    def __init__(self, kb_headlines, wiki_cache=None, device="cpu", allow_live_wiki=False):
        from rank_bm25 import BM25Okapi

        self.pre = LegacyPreprocessor()
        self.corpus = [self.pre.clean_text(h) for h in kb_headlines]
        self.bm25 = BM25Okapi([self.pre.bm25_tokens(doc) for doc in self.corpus])
        self.wiki = wiki_cache
        self.allow_live_wiki = allow_live_wiki
        self.device = device
        self._model = None

    def _load_gpt2(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device).eval()
        self.bad_words_ids = [self.tokenizer.encode(w, add_special_tokens=False) for w in GPT2_BAD_WORDS]
        self._torch = torch

    def retrieve(self, headline: str, k: int = 3):
        scores = self.bm25.get_scores(self.pre.bm25_tokens(headline))
        top = np.argsort(scores)[::-1][:k]
        docs = [self.corpus[i] for i in top]
        doc_scores = [float(scores[i]) for i in top]
        if self.wiki is not None:
            result = self.wiki.get(headline, allow_live=self.allow_live_wiki)
            if result["status"] in wiki_mod.LEGACY_SCORES:
                docs.append(result["text"])
                doc_scores.append(wiki_mod.LEGACY_SCORES[result["status"]])
        return docs, doc_scores

    def generate(self, prompt: str, score: float, seed: int):
        if self._model is None:
            self._load_gpt2()
        torch = self._torch
        torch.manual_seed(seed)
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            out = self._model.generate(
                inputs, max_new_tokens=100, min_new_tokens=20, temperature=0.7, repetition_penalty=2.0,
                num_beams=2, do_sample=True, early_stopping=False, bad_words_ids=self.bad_words_ids,
                pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id)
        full = self.tokenizer.decode(out[0], skip_special_tokens=True)
        generated = full[len(prompt):].strip()
        verdict, explanation = parse_generated_verdict(generated, score=score)
        return verdict, explanation, generated

    def predict(self, headline: str, seed: int = 0) -> dict:
        docs, scores = self.retrieve(headline)
        _, _, _, normalized, _ = apply_rules("Fake", "", headline, scores)
        gpt2_verdict, explanation, generated = self.generate(build_prompt(headline, docs), normalized, seed)
        verdict, explanation, rule, normalized, wiki_present = apply_rules(gpt2_verdict, explanation, headline, scores)
        return {"verdict": verdict, "gpt2_verdict": gpt2_verdict, "rule": rule, "explanation": explanation,
                "relevance_score": round(normalized, 3), "wikipedia_used": wiki_present, "generated": generated}
