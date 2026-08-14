from sentence_transformers import SentenceTransformer
import faiss, numpy as np
from datasets import load_dataset
import re

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


ds = load_dataset("json", data_files="datasets-issues-with-comments.jsonl", split="train")  # adjust name/split

ds = ds.map(combine_fields)
print(ds[0])