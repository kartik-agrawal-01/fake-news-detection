"""Turning model output into a Real / Fake verdict and a short explanation.

``parse_generated_verdict`` is the parser from the original GPT-2 pipeline, unchanged in behaviour but pulled out
as a pure function so it can be tested. ``explain`` builds the explanation for the fine-tuned classifier from its
confidence and the retrieved evidence (no text generation involved).
"""
import re

FAKE_WORDS = ["fake", "false", "no evidence", "conspiracy", "baseless", "untrue", "hoax", "not", "no",
              "contradicts", "failed", "accused", "youtube", "http", "source:"]
REAL_WORDS = ["real", "true", "confirmed", "matches", "supported", "verified", "is true", "signed", "passed", "law",
              "bill", "implemented", "president", "election", "attack", "war"]


def clean_generated_text(text: str) -> str:
    """Remove repeated boilerplate, underscores and links from GPT-2 output."""
    for prefix in (r"Fact:", r"Source:", r"The truth is"):
        text = re.sub(rf"(\b{prefix}\s*[^.]*?\.)\s*\1+", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"_{2,}", " ", text)
    return re.sub(r"(http[^ \n]+)", "[Source Link]", text)


def parse_generated_verdict(generated_text: str, score: float | None = None) -> tuple[str, str]:
    """(verdict, explanation) from free text: an explicit "Verdict: X" wins, otherwise count keywords."""
    if not generated_text.strip():
        return "Fake", "No supporting evidence found in facts; treated as unverified."
    text = clean_generated_text(generated_text.strip())

    match = re.search(r"(?i)(verdict:\s*)(fake|real|true|false)", text)
    if match:
        verdict = "Real" if match.group(2).lower() in ("true", "real") else "Fake"
        explanation = text[match.end():match.end() + 100].strip()
        if len(explanation) < 10:
            explanation = "Analysis based on available facts."
        return verdict, explanation

    # Substring counts, exactly as in the original (so "not" also counts inside "nothing", etc.).
    lower = text.lower()
    fake_signals = sum(lower.count(w) for w in FAKE_WORDS)
    real_signals = sum(lower.count(w) for w in REAL_WORDS)
    if score is not None and score >= 5.0 and real_signals > fake_signals:
        return "Real", f"Supported by high-relevance facts (score: {score:.1f}, keywords: {real_signals}). {text[:80]}..."
    if real_signals > fake_signals:
        return "Real", f"Supported by facts (keywords: {real_signals}). {text[:80]}..."
    if fake_signals > 1:
        return "Fake", f"Lacks evidence (keywords: {fake_signals}). {text[:80]}..."
    return "Fake", f"Lacks supporting facts (keywords: {fake_signals}). {text[:80]}..."


def confidence_phrase(p: float) -> str:
    if p >= 0.9:
        return "high"
    if p >= 0.7:
        return "moderate"
    return "low"


def explain(prob_real: float, hits, wiki_text: str | None = None, evidence_used: bool = True) -> tuple[str, str]:
    """(verdict, explanation) for the classifier. ``hits`` are retrieval.Hit objects (may be empty).

    ``evidence_used`` says whether the model read the hits (headline + evidence model) or only the headline.
    """
    verdict = "Real" if prob_real >= 0.5 else "Fake"
    p = prob_real if verdict == "Real" else 1 - prob_real
    lines = [f"The classifier rates this headline **{verdict}** with {p:.0%} probability "
             f"({confidence_phrase(p)} confidence)."]
    if hits:
        best = hits[0]
        role = (f"{len(hits)} match(es) were passed to the model as evidence" if evidence_used
                else "the model itself reads only the headline")
        lines.append(f"The closest verified-real PolitiFact headline in the knowledge base is \"{best.text}\" "
                     f"(BM25 score {best.score:.1f}); {role}.")
    else:
        lines.append("No headline in the knowledge base shares any keywords with it"
                     + (", so the model saw no evidence." if evidence_used else "."))
    if wiki_text:
        lines.append(f"Wikipedia context: \"{wiki_text[:200]}\"")
    lines.append("The model judges wording and similarity to known headlines. It does not verify facts, so treat "
                 "this as a hint, not a fact-check.")
    return verdict, " ".join(lines)
