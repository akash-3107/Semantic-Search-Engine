# Semantic-Search-Engine

# Semantic Search Engine

A semantic search engine for GitHub issues. Encodes issue text with `sentence-transformers`, indexes the embeddings with FAISS, and compares retrieval quality against a TF-IDF baseline.

## How it works

The pipeline (all in `main.py`) runs in five stages:

1. **Load & clean** — Loads `datasets-issues-with-comments.jsonl` via HuggingFace `datasets`, combines each issue's `title` + `body` into a single `text` field, and strips markdown/HTML/URLs from it. Empty or near-empty issues (< 20 chars) are dropped.
2. **Encode** — Embeds every issue with `sentence-transformers/all-MiniLM-L6-v2` (384-dim), batched for speed.
3. **Index** — Builds a FAISS `IndexFlatL2` index over the embeddings. The index and the id → title/text/url metadata are cached to disk (`github_issues.index`, `corpus_metadata.pkl`) so re-runs skip re-embedding.
4. **Search** — `search(query, model, index, meta, top_k)` embeds a query with the same model and returns the top-K nearest issues by L2 distance.
5. **Baseline comparison** — A TF-IDF + cosine-similarity retriever (scikit-learn) runs the same test queries side by side with the semantic search, so you can eyeball where each approach wins.

## Requirements

```bash
pip install sentence-transformers faiss-cpu datasets scikit-learn numpy
```

## Usage

1. Place `datasets-issues-with-comments.jsonl` in the repo root (already included).
2. Run:

```bash
python main.py
```

On first run this will encode the full corpus and build `github_issues.index` + `corpus_metadata.pkl`. Subsequent runs reuse the cached index instead of re-embedding.

3. To query the index directly, import and call `search`:

```python
results = search("memory leak when using DataLoader with num_workers > 0", model, index, meta, top_k=5)
for r in results:
    print(f"[{r['rank']}] score={r['score']:.4f} — {r['title']}")
```

Note: `IndexFlatL2` returns L2 **distance**, not similarity — lower scores mean a closer match.

## Evaluating retrieval quality

`main.py` includes a `test_queries` list and prints the top-3 results from both the semantic (MiniLM) and TF-IDF retrievers for each query, so you can compare them manually. Semantic search should do better on paraphrased or symptom-style queries (different wording than the original issue text); TF-IDF can still win on queries with exact rare keywords (specific error codes, function names).

## Files

| File | Purpose |
|---|---|
| `main.py` | Full pipeline: load, clean, encode, index, search, TF-IDF comparison |
| `datasets-issues-with-comments.jsonl` | Source corpus of GitHub issues |
| `github_issues.index` | Cached FAISS index (generated on first run) |
| `corpus_metadata.pkl` | Cached id → title/text/url mapping (generated on first run) |

## Roadmap

- [ ] Quantitative evaluation via NDCG against manually labeled relevance judgments
