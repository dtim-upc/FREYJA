import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from collections import defaultdict

class LSHIndex:
    def __init__(self, dim, num_hashes=16, num_tables=4, multiprobe=False):
        """
        dim: dimension of embeddings
        num_hashes: number of random hyperplanes per table
        num_tables: number of independent hash tables
        multiprobe: whether to probe nearby buckets (Hamming distance 1)
        """
        self.dim = dim
        self.num_hashes = num_hashes
        self.num_tables = num_tables
        self.multiprobe = multiprobe

        # One set of random hyperplanes per table
        self.hyperplanes = [np.random.randn(num_hashes, dim) for _ in range(num_tables)]
        self.buckets = [defaultdict(list) for _ in range(num_tables)]

    def _simhash(self, vec, table_idx):
        """Project onto hyperplanes → binary code"""
        projections = np.dot(self.hyperplanes[table_idx], vec)
        return tuple((projections > 0).astype(int))

    def _neighbors(self, h):
        """Generate nearby buckets (Hamming distance 1) if multiprobe is enabled"""
        if not self.multiprobe:
            return [h]
        neighbors = [h]
        for i in range(len(h)):
            flipped = list(h)
            flipped[i] = 1 - flipped[i]  # flip bit
            neighbors.append(tuple(flipped))
        return neighbors

    def add(self, vec, idx):
        """Insert vector with identifier (normalize first)"""
        vec = vec / np.linalg.norm(vec)
        for t in range(self.num_tables):
            h = self._simhash(vec, t)
            self.buckets[t][h].append(idx)

    def query(self, vec, top_k=5, embeddings=None):
        """Return top_k most similar embeddings to vec"""
        vec = vec / np.linalg.norm(vec)

        candidate_ids = set()
        for t in range(self.num_tables):
            h = self._simhash(vec, t)
            for neighbor in self._neighbors(h):
                candidate_ids.update(self.buckets[t].get(neighbor, []))

        if not candidate_ids:
            return []

        # Vectorized cosine similarity (dot product since normalized)
        cands = embeddings[list(candidate_ids)]
        sims = np.dot(cands, vec)  # dot product = cosine similarity
        idxs = list(candidate_ids)

        # Take top-k
        topk_idx = np.argsort(-sims)[:top_k]
        return [(idxs[i], sims[i]) for i in topk_idx]
    

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
        match = metadata[(metadata['table'].str.strip() == dataset.strip()) & (metadata['column'].str.strip() == attribute.strip())]
        if not match.empty:
            query_index = match['id'].iloc[0]  # get the first match
        else:
            raise Exception("Query attribute not found")

        query_vec = embeddings[query_index].reshape(1, -1)

        # Perform search
        results = index.query(query_vec.flatten(), top_k=k, embeddings=embeddings)

        results_list = []
        for _, (idx, sim) in enumerate(results):
            meta = metadata.iloc[idx]
            results_list.append({
                "candidate_ds": meta['table'],
                "candidate_attr": meta['column'],
                "distance": sim  # or 1 - sim if you want it as a "distance"
            })
        results = pd.DataFrame(results_list)

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

datalake_name = "santos_small"
embeddings = np.load(f"embeddings/{datalake_name}/embeddings.npy")
metadata = pd.read_csv(f"embeddings/{datalake_name}/column_metadata.csv")
ground_truth_path = f'C:/Projects/{datalake_name}/{datalake_name}_ground_truth.csv'
k = 10
step = 1

# Normalize embeddings once up front and build index
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
index = LSHIndex(dim=embeddings.shape[1], num_hashes=16, num_tables=8, multiprobe=True)
for i, emb in enumerate(embeddings):
    index.add(emb, i)

print("✅ LSH index built.")

# # Example query
# query_idx = 0  # pick first column
# query_vec = embeddings[query_idx]

# results = index.query(query_vec, top_k=10, embeddings=embeddings)

# print("Query:", metadata.iloc[query_idx].to_dict())
# for idx, sim in results:
#     print(f"Match: {metadata.iloc[idx].to_dict()} | Cosine={sim:.4f}")

compute_and_evaluate_ranking(k, step, ground_truth_path, metadata, embeddings, index)