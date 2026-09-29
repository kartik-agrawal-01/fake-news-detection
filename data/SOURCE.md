# Data sources

## PolitiFact headlines: `politifact_fake.csv`, `politifact_real.csv`

From **FakeNewsNet** (https://github.com/KaiDMML/FakeNewsNet), the PolitiFact part of its minimal dataset:
432 fake and 624 real news articles, one row each, with columns `id`, `news_url`, `title` and `tweet_ids`.
Labels come from PolitiFact fact-checks. This project uses only the `title` column.

Please cite:

> Kai Shu, Deepak Mahudeswaran, Suhang Wang, Dongwon Lee and Huan Liu. *FakeNewsNet: A Data Repository with News
> Content, Social Context and Spatiotemporal Information for Studying Fake News on Social Media.* arXiv:1809.01286, 2018.

> Kai Shu, Amy Sliva, Suhang Wang, Jiliang Tang and Huan Liu. *Fake News Detection on Social Media: A Data Mining
> Perspective.* ACM SIGKDD Explorations Newsletter 19(1), 2017.

Things to know about this data:
- 73 of the 1,056 rows repeat another row's title exactly; `src/prepare_data.py` groups them so duplicates never
  straddle a train/test split. Two duplicated titles carry both labels.
- Some "titles" are page titles rather than headlines (e.g. "Account Suspended", "Browse Congressional Bills"), and
  a few are not in English.

## Wikipedia cache: `wikipedia_cache.json`

Two-sentence Wikipedia summaries looked up for each headline by `python -m src.wiki` (September 2026), stored so
that evaluation does not depend on live Wikipedia. Wikipedia text is licensed under CC BY-SA 4.0
(https://en.wikipedia.org/wiki/Wikipedia:Text_of_the_Creative_Commons_Attribution-ShareAlike_4.0_International_License).

## Processed data

`data/processed/headlines.csv` (headline, label, duplicate group, CV fold, original split) is not committed.
`python -m src.prepare_data` regenerates it.
