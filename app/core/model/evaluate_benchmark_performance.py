from dataclasses import dataclass
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

import pandas as pd
from tqdm import tqdm

import joblib

import time

@dataclass
class ModelExecutionConfig:
    """Configuration for the execution of the model."""
    k: int
    step: int
    ground_truth_path: Path
    distances_folder_path: Path
    model_path: Path


class ModelExecution:
    """Main orchestrator to execute the model."""
    def __init__(self, config: ModelExecutionConfig = None):
        self.config = config
        self.model = joblib.load(self.config.model_path)

    @staticmethod
    def obtain_ranking(model, distances_path, top_k, print_results = True) -> pd.DataFrame:
        """Obtains the top-k joinable columns from a datalake, based on the model predictions"""
        st = time.time()
        try:
            # Read the distances and do some preprocessing
            distances: pd.DataFrame = pd.read_csv(distances_path, header = 0, encoding='latin1', on_bad_lines="skip")

            dataset_names = distances["dataset_name_2"] # We store dataset and attribute names to later identify the ranking
            attribute_names = distances["attribute_name_2"]

            distances = distances.drop(columns=['dataset_name', 'dataset_name_2', 'attribute_name', 'attribute_name_2'], axis=1)
            distances = distances[model.feature_names_in_] 

            # Use the model to predict (preventing some weird lines that might have slipped in)
            distances_numeric = distances.apply(pd.to_numeric, errors='coerce') # Convert everything to float, invalid parsing becomes NaN
            valid_rows = distances_numeric.dropna(axis=0, how='any') # Keep track of valid rows
            predicted_scores = model.predict(valid_rows) # Predict only on valid rows
            distances.loc[valid_rows.index, "predicted_score"] = predicted_scores # Assign predictions back only to the valid rows

            distances["target_ds"] = dataset_names
            distances["target_attr"] = attribute_names
            distances["normalized_score"] = distances["predicted_score"] / 0.5

            total_time = (time.time() - st)
            top_k_joins = distances.sort_values(by='normalized_score', ascending=False).head(top_k)[["normalized_score", "target_ds", "target_attr"]]
            
            if print_results:
                logger.info(f"Evaluation time -> {total_time}")
                logger.info(top_k_joins)

        except Exception as e:
            logger.error(e)

        return top_k_joins


    def evaluate_benchmark(self):
        """Obtains precision, recall and MAP scores for a given join discovery benchmark"""
        try:
            # Read the ground truth and obtain, for every target column, the amount of candidate columns that it has a join with. This will allow us to calculate the recall,
            # as it indicates the maximum possible joins, regardless of the value of k
            ground_truth = pd.read_csv(self.config.ground_truth_path, header = 0)
            pair_counts = ground_truth.groupby(['target_ds', 'target_attr']).size().reset_index(name='joins_count')

            # Initialize the matrix of metrics
            num_observations = int(self.config.k / self.config.step)
            precision = [0] * num_observations
            recall = [0] * num_observations
            max_recall = [0] * num_observations
            MAP = [0] * num_observations

            total_time = 0 # Initialize execution time

            for _, row in tqdm(pair_counts.iterrows(), total=len(pair_counts)):
                dataset: str = row['target_ds']
                attribute: str = row['target_attr']
                count = row['joins_count']

                start_time = time.time()

                distances_path = self.config.distances_folder_path / f"distances_{dataset.replace('.csv', '_profile')}_{attribute.replace('/', '_').replace(': ','_')}.csv"
                top_k_joins: pd.DataFrame = self.obtain_ranking(self.model, distances_path, self.config.k, print_results=False)

                total_time += (time.time() - start_time) # When tracking the time, we just consider the obtention of the distances

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

                    top_k_joins_iter = top_k_joins.head(k_iter * self.config.step)

                    for position in top_k_joins_iter.itertuples(index=False):
                        pair = (position.target_ds, position.target_attr)
                        if pair in valid_pairs: 
                            count_sem += 1
                            ap += count_sem / (count_positions + 1)
                        count_positions += 1


                    precision[k_iter - 1] += count_sem / (k_iter * self.config.step)
                    if count_sem != 0:
                        MAP[k_iter - 1] += ap / count_sem
                    recall[k_iter - 1] += count_sem / count
                    max_recall[k_iter - 1] += (k_iter * self.config.step) / count

            logger.info(f"TOTAL time needed to load the distances and execute the model: {total_time:.4f}")
            logger.info(f"AVERAGE time needed to load the distances and execute the model: {total_time / len(pair_counts):.4f}")

            precision_scores = [round(p / len(pair_counts), 4) for p in precision]
            raw_recall_scores = [round(r / len(pair_counts), 4) for r in recall]
            maximum_recall_scores = [round(mr / len(pair_counts), 4) for mr in max_recall]
            normalized_recall_scores = [round((rn / len(pair_counts)) / (mr / len(pair_counts)), 4) for rn, mr in zip(recall, max_recall)]
            map_scores = [round(m / len(pair_counts), 4) for m in MAP]

            logger.info("Precisions:", precision_scores)
            logger.info("Recall (raw):", raw_recall_scores)
            logger.info("Maximum recall:", maximum_recall_scores)
            logger.info("Recall (normalized):", normalized_recall_scores)
            logger.info("MAP:", map_scores)
        
        except Exception as e:
            logger.error(e)

        return {
            "precision": precision_scores,
            "recall": normalized_recall_scores,
            "map": map_scores
        }