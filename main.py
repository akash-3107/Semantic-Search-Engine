from sentence_transformers import SentenceTransformer
import faiss, numpy as np
from datasets import load_dataset
import re, time



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
print(embeddings.shape)