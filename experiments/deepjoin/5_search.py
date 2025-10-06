import json
import numpy as np
import faiss
import time
import pandas as pd
from tqdm import tqdm

def load_metadata(meta_path):
    meta = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta


def build_hnsw_index(embeddings, m=32, ef_search=64):
    """Pure HNSW index (paper's default for large table repos)."""
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, m)
    index.hnsw.efSearch = ef_search
    print(f"Training HNSW index with m={m}, ef_search={ef_search}...")
    index.add(embeddings)
    return index


def build_ivfpq_hnsw_index(embeddings, nlist=1000, pq_m=16, pq_bits=8, ef_search=64):
    """
    IVFPQ + HNSW over coarse quantizer (paper's billion-scale setup).
    - nlist: number of IVF clusters
    - pq_m: number of PQ subvectors
    - pq_bits: bits per PQ code
    """
    dim = embeddings.shape[1]

    # Coarse quantizer is HNSWFlat
    quantizer = faiss.IndexHNSWFlat(dim, 32)
    quantizer.hnsw.efSearch = ef_search

    # IVFPQ index
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, pq_bits)

    print(f"Training IVFPQ+HNSW index (nlist={nlist}, pq_m={pq_m}, pq_bits={pq_bits})...")
    index.train(embeddings)
    index.add(embeddings)
    return index


def compute_and_evaluate_ranking(k, step, ground_truth_path, metadata, embeddings, index):
    # Read the ground truth and obtain, for every target column, the amount of candidate columns that it has a join with. This will allow us to calculate the recall,
    # as it indicates the maximum possible joins, regardless of the value of k
    ground_truth = pd.read_csv(ground_truth_path, header = 0)
    pair_counts = ground_truth.groupby(['target_ds', 'target_attr']).size().reset_index(name='joins_count')

    # Initialize the matrix of metrics
    num_observations = int(k / step)
    precision = [0] * num_observations
    recall = [0] * num_observations
    max_recall = [0] * num_observations
    MAP = [0] * num_observations

    # Initialize execution time
    total_time = 0

    for _, row in tqdm(pair_counts.iterrows(), total=len(pair_counts)):
        dataset = row['target_ds']
        attribute = row['target_attr']
        count = row['joins_count']

        st = time.time()

        # Find query vector
        query_index = None
        for line in metadata:
            if line["table"].strip() == dataset and line["column"].strip() == attribute:
                query_index = line["id"]
                break

        query_vec = embeddings[query_index].reshape(1, -1)

        # Perform search
        D, I = index.search(query_vec, k)

        results = []
        for _, (idx, dist) in enumerate(zip(I[0], D[0])):
            meta = metadata[idx]
            # print(f"{rank+1}. {meta['table']}.{meta['column']}  (dist={dist:.4f})")
            results.append({
                "candidate_ds": meta['table'],
                "candidate_attr": meta['column'],
                "distance": dist
            })
        results = pd.DataFrame(results)

        total_time += (time.time() - st)

        # Precompute a lookup set of valid (candidate_ds, candidate_attr) for this query
        valid_pairs = set(
            ground_truth.loc[
                (ground_truth['target_ds'] == dataset) &
                (ground_truth['target_attr'] == attribute),
                ['candidate_ds', 'candidate_attr']
            ].itertuples(index=False, name=None)
        )

        # For every k that we want to assess the ranking of, we get the top k joins and check how many appear in the grpund truth
        for k_iter in range(1, num_observations + 1):
            count_sem = 0
            ap = 0
            count_positions = 0

            for position in results.head(k_iter * step).itertuples(index=False):
                pair = (position.candidate_ds, position.candidate_attr)
                if pair in valid_pairs: 
                    count_sem += 1
                    ap += count_sem / (count_positions + 1)
                count_positions += 1

            precision[k_iter - 1] += count_sem / (k_iter * step)
            if count_sem != 0:
                MAP[k_iter - 1] += ap / count_sem
            recall[k_iter - 1] += count_sem / count
            max_recall[k_iter - 1] += (k_iter * step) / count

    print("AVERAGE time to load the distances and execute the model:")
    print("----%.2f----" % (total_time / len(pair_counts)))

    print("Precisions:", [round(element / len(pair_counts), 4) for element in precision])
    print("Recall:", [round(element / len(pair_counts), 4) for element in recall])
    print("Max recall:", [round(element / len(pair_counts), 4) for element in max_recall])
    print("Recall percentage:", [round((recall_iter / len(pair_counts)) / (max_recall_iter / len(pair_counts)), 4) for recall_iter, max_recall_iter in zip(recall, max_recall)])
    print("MAP:", [round(element / len(pair_counts), 4) for element in MAP])

    return [round(element / len(pair_counts), 4) for element in precision]


benchmark_name = "omnimatch_culture_recreation"
embeddings = np.load(f"embeddings/{benchmark_name}/embeddings.npy").astype("float32")
metadata = load_metadata(f"embeddings/{benchmark_name}/meta.jsonl")
ground_truth_path = '../benchmarks/omnimatch_culture_recreation/omnimatch_culture_recreation_ground_truth.csv'
k = 30
step = 5

mode = "hnsw" # Index mode: "hnsw" or "ivfpq"

# HNSW params
hnsw_m = 32 # The number of neighbors for HNSW. This is typically 32
hnsw_ef_search = 64

# IVFPQ params
nlist = 1000  # The number of cells (space partition). Typical value is sqrt(N), with N being the number of vectors that you’re going to store in the index for search (aka number of columns).
pq_m = 16 # The number of sub-vector. Typically this is 8, 16, 32, etc.
pq_bits = 8 # bits per sub-vector. This is typically 8, so that each sub-vec is encoded by 1 byte

# Build chosen index
if mode == "hnsw":
    index = build_hnsw_index(embeddings, m=hnsw_m, ef_search=hnsw_ef_search)
else:
    index = build_ivfpq_hnsw_index(embeddings, nlist=nlist, pq_m=pq_m, pq_bits=pq_bits, ef_search=hnsw_ef_search)

# Use GPU if available
if faiss.get_num_gpus() > 0:
    index = faiss.index_cpu_to_all_gpus(index)

compute_and_evaluate_ranking(k, step, ground_truth_path, metadata, embeddings, index)
