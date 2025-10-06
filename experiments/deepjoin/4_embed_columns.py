import os
import json
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device

def column_to_text(column, title=None):
    """
    Generate descriptive text for a column.
    Format: Title. colname contains N values (max, min, avg): val1, val2, ...
    """
    colname = column.name
    cells = column.dropna().astype(str)
    n_unique = len(cells.unique())
    word_counts = cells.apply(lambda x: len(x.split()))
    
    max_len = word_counts.max() if len(word_counts) > 0 else 0
    min_len = word_counts.min() if len(word_counts) > 0 else 0
    avg_len = round(word_counts.mean(), 2) if len(word_counts) > 0 else 0
    
    unique_vals_text = ", ".join(cells.unique())
    text = f"{title}. {colname} contains {n_unique} values ({max_len}, {min_len}, {avg_len}): {unique_vals_text}."
    return text

def process_file(file_path, title_prefix, start_idx=0):
    """
    Convert all columns in a CSV file to descriptive text.
    Returns a list of texts and metadata.
    """
    df = pd.read_csv(file_path, encoding="latin1")
    texts, meta = [], []
    idx = start_idx
    for col in df.columns:
        text = column_to_text(df[col], title=f"{title_prefix}")
        texts.append(text)
        meta.append({"id": idx, "table": os.path.basename(file_path), "column": col, "text": text})
        idx += 1
    return texts, meta, idx

def encode_columns(model, texts, batch_size=256):
    """
    Encode a list of column texts using SentenceTransformer.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    return embeddings.astype("float32")

def save_outputs(output_dir, embeddings, meta):
    """
    Save embeddings, metadata, and CSV summary.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    
    meta_path = os.path.join(output_dir, "meta.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    
    pd.DataFrame(meta).to_csv(os.path.join(output_dir, "columns_texts.csv"), index=False)
    print(f"Wrote embeddings and metadata to {output_dir}")

benchmark_name = "omnimatch_culture_recreation"
benchmark_folder = f"../benchmarks/{benchmark_name}/datalake"
model_path = f"models/{benchmark_name}"
output_dir = f"embeddings/{benchmark_name}"

device = get_device()
model = SentenceTransformer(model_path, device=device)

all_texts, all_meta = [], []
idx = 0
print("Transforming columns to text...")
for file_name in tqdm(os.listdir(benchmark_folder)):
    file_path = os.path.join(benchmark_folder, file_name)
    texts, meta, idx = process_file(file_path, title_prefix=f"{file_name} information", start_idx=idx)
    all_texts.extend(texts)
    all_meta.extend(meta)

print("Encoding columns...")
embeddings = encode_columns(model, all_texts, batch_size=256)

save_outputs(output_dir, embeddings, all_meta)