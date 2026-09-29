"""Wikipedia lookups with an on-disk cache, so evaluation numbers are reproducible.

The lookup logic is the one from the original retrieval module: clean the headline, special-case
"X won the <year> ... election" claims, ask Wikipedia for a 2-sentence summary, and on a disambiguation page
fall back to the first search result. Every result is stored in data/wikipedia_cache.json keyed by headline.

Evaluation only reads the cache (``allow_live=False``); run ``python -m src.wiki`` once to fill it.
The app reads the cache first and looks up new headlines live.

Wikipedia text is CC BY-SA 4.0 (https://en.wikipedia.org/wiki/Wikipedia:Copyrights).
"""
import argparse
import json
import re
import time

from . import config

USER_AGENT = "fake-news-detection-portfolio/1.0 (https://github.com/kartik-agrawal-01/fake-news-detection)"

# Status of a lookup and the relevance score the original pipeline gave it.
SUMMARY, FALLBACK, PLACEHOLDER, NOTHING = "summary", "fallback", "placeholder", "nothing"
LEGACY_SCORES = {SUMMARY: 5.0, FALLBACK: 4.0, PLACEHOLDER: 2.0}
PLACEHOLDER_TEXT = "No additional dynamic fact available – relying on KB."


def parse_query(headline: str) -> str:
    """Turn a headline into a Wikipedia query (the original rule for election claims included)."""
    clean = re.sub(r"[^a-zA-Z0-9\s]", " ", headline).strip().lower()
    if "won the" in clean and "election" in clean:
        match = re.search(r"(\d{4}\s+[a-z\s]+election)", clean)
        if match:
            return match.group(1).strip()
    return clean


class TransientError(RuntimeError):
    """Network trouble: the lookup should be retried rather than cached."""


def lookup_live(headline: str) -> dict:
    """One live lookup, following the original two-attempt logic. Returns {"status", "text", "query"}."""
    import requests
    import wikipedia

    wikipedia.set_lang("en")
    wikipedia.set_user_agent(USER_AGENT)
    query = parse_query(headline)
    result = {"status": NOTHING, "text": None, "query": query}
    for attempt in range(2):
        try:
            return {"status": SUMMARY, "text": wikipedia.summary(query, sentences=2), "query": query}
        except wikipedia.exceptions.DisambiguationError:
            titles = wikipedia.search(query)
            if titles:
                try:
                    page = wikipedia.page(titles[0])
                except (requests.RequestException, ValueError) as e:
                    raise TransientError(str(e)) from e
                except Exception:
                    if attempt == 1:
                        result = {"status": PLACEHOLDER, "text": PLACEHOLDER_TEXT, "query": query}
                    continue
                return {"status": FALLBACK, "text": page.summary[:200], "query": query}
        except (requests.RequestException, ValueError) as e:
            # ValueError covers JSON decode errors from throttled or failed HTTP responses.
            raise TransientError(str(e)) from e
        except Exception:
            if attempt == 1:
                result = {"status": PLACEHOLDER, "text": PLACEHOLDER_TEXT, "query": query}
    return result


class WikiCache:
    def __init__(self, path=config.WIKI_CACHE):
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def get(self, headline: str, allow_live: bool = False):
        """Cached result for ``headline``; looks it up live when missing and ``allow_live`` is set."""
        if headline in self.data:
            return self.data[headline]
        if not allow_live:
            raise KeyError(f"No cached Wikipedia result for {headline!r}. Run `python -m src.wiki` first.")
        result = lookup_live(headline)
        self.data[headline] = result
        return result

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ordered = dict(sorted(self.data.items()))
        self.path.write_text(json.dumps(ordered, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description="Fill the Wikipedia cache for every headline in the dataset.")
    parser.add_argument("--delay", type=float, default=0.5, help="seconds between requests")
    args = parser.parse_args()

    headlines = pd.read_csv(config.PROCESSED)["headline"].fillna("").astype(str).unique()
    cache = WikiCache()
    todo = [h for h in headlines if h not in cache.data]
    print(f"{len(headlines)} headlines, {len(todo)} not cached yet")
    for i, headline in enumerate(todo, 1):
        for wait in (5, 30, 120, None):
            try:
                cache.get(headline, allow_live=True)
                break
            except TransientError as e:
                if wait is None:
                    print(f"  giving up on {headline!r}: {e}")
                    break
                print(f"  network error ({e}); retrying in {wait}s")
                time.sleep(wait)
        if i % 25 == 0 or i == len(todo):
            cache.save()
            print(f"[{i}/{len(todo)}] cached", flush=True)
        time.sleep(args.delay)
    cache.save()
    counts = pd.Series([cache.data[h]["status"] for h in headlines if h in cache.data]).value_counts()
    print(counts.to_string())


if __name__ == "__main__":
    main()
