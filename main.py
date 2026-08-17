from sentence_transformers import SentenceTransformer
import faiss, numpy as np
from datasets import load_dataset
import re, time
import os
import pickle

#----------------------------------------Data load and pre-process----------------------------------------------

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)   # strip code blocks
    text = re.sub(r"<[^>]+>", " ", text)                       # strip HTML tags
    text = re.sub(r"[#*_`>]+", " ", text)                      # strip markdown symbols
    text = re.sub(r"http\S+", " ", text)                       # strip URLs
    text = re.sub(r"\s+", " ", text).strip()
    return text
    
def combine_fields(example):
    title = clean_text(example.get("title", ""))
    body = clean_text(example.get("body", ""))
    example["text"] = f"{title}. {body}".strip()
    return example



# Load the dataset from downloaded file from HF
ds = load_dataset("json", data_files="datasets-issues-with-comments.jsonl", split="train")  # adjust name/split

ds = ds.map(combine_fields)
#print(ds[0])

ds = ds.filter(lambda ex: len(ex["text"]) > 20)   # drop empty/near-empty issues

# Keep an id -> text mapping for later lookup
corpus_ids = list(range(len(ds)))
corpus_texts = ds["text"]

#----------------------------------------Encoding----------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")         # CPU/GPU detected automatically

def encode_corpus(texts, model, batch_size=128):
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar = True,
        convert_to_numpy = True,
        normalize_embeddings = False       # Let FAISS handle L2 distance clearly
    )

    print(f"Encoded {len(texts)} docs in {time.time() - start:.1f}s")
    return embeddings.astype("float32")

embeddings = encode_corpus(corpus_texts, model)
#print(embeddings.shape)

#----------------------------------------FAISS Indexing----------------------------------------------

dim = embeddings.shape[1]  # 384

if not os.path.isfile("github_issues.index"):
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"Indexed {index.ntotal} vectors")
    # Persist to disk
    faiss.write_index(index, "github_issues.index")

index = faiss.read_index("github_issues.index")

# Persist the id -> text/metadata mapping separately (FAISS only stores vectors)
if not os.path.isfile("corpus_matadata.pkl"):
    with open("corpus_metadata.pkl", "wb") as f:
        pickle.dump(
            {"ids": corpus_ids, "texts": corpus_texts, "titles": ds["title"], "urls": ds["url"]},
            f,
        )

with open("corpus_metadata.pkl", "rb") as f:
    meta = pickle.load(f)


#----------------------------------------Query Function (Search)----------------------------------------------

def search(query: str, model, index, meta, top_k: int = 5):
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vec, top_k)

    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        if idx == -1:
            continue
        results.append({
            "rank": rank + 1,
            "score": float(dist),          # L2 distance — lower is more similar
            "title": meta["titles"][idx],
            "text": meta["texts"][idx][:300],
            "url": meta["urls"][idx] if meta.get("urls") else None,
        })
    return results

"""
# Example usage
results = search("memory leak when using DataLoader with num_workers > 0", model, index, meta, top_k=5)
for r in results:
    print(f"[{r['rank']}] score={r['score']:.4f} — {r['title']}")
"""


#----------------------------------------TFIDF comparison----------------------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf_vectorizer = TfidfVectorizer(max_features=50000, stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_texts)

def tfidf_search(query: str, top_k: int = 5):
    query_vec = tfidf_vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    return [
        {"rank": i + 1, "score": float(sims[idx]), "title": ds["title"][idx], "text": corpus_texts[idx][:300]}
        for i, idx in enumerate(top_idx)
    ]

test_queries = [
    "segmentation fault on model.eval()",
    "cannot install package on M1 mac",
    "docs missing example for custom dataset",
    # ... 20-30 total, ideally covering different issue "types":
    #     bug reports, feature requests, doc gaps, install errors, perf issues
]

for q in test_queries:
    print(f"\n=== Query: {q} ===")
    print("-- Semantic (MiniLM) --")
    for r in search(q, model, index, meta, top_k=3):
        print(f"  {r['score']:.3f}  {r['title']}")
    print("-- TF-IDF --")
    for r in tfidf_search(q, top_k=3):
        print(f"  {r['score']:.3f}  {r['title']}")