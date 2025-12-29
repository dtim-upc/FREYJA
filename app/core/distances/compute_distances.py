from dataclasses import dataclass
import logging
logger = logging.getLogger(__name__)

import time
import pandas as pd
import numpy as np
import Levenshtein

from pathlib import Path
from typing import Any, Dict
from tqdm import tqdm

from app.core.profiling.metrics import MetricProperties

@dataclass
class ComputeDistancesConfig:
    """Configuration for the data profiler."""
    profiles_file_path: Path
    ground_truth_path: Path
    output_distances_path: Path


class ComputeDistances:
    """Main orchestrator to compute distances."""
    
    def __init__(self, config: ComputeDistancesConfig):
        self.config = config
        self.query_dataset = None
        self.query_attribute = None
        self.profiles: pd.DataFrame = pd.read_pickle(self.config.profiles_file_path)
        self.distance_patterns = MetricProperties().distance_patterns
        if config.ground_truth_path is not None:
            self.ground_truth = pd.read_csv(self.config.ground_truth_path)
        
        # Create necessary directories if needed
        if self.config.output_distances_path.suffix.lower() == "":
            self.config.output_distances_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def levenshtein_distance(a, b) -> int:
        """Calculate Levenshtein distance between two values."""
        if a is None and b is None:
            return float("inf")
        if a is None:
            return len(b)
        if b is None:
            return len(a)
        return Levenshtein.distance(str(a), str(b))

    def _compute_distance_value(self, query_value: Any, target_value: Any, pattern: str) -> float:
        """Compute distance between two values based on the specified pattern."""
        if pattern == "substraction":
            if pd.isna(target_value) or pd.isna(query_value):
                return 0
            return query_value - target_value
            
        elif pattern == "containment":
            if query_value and target_value:
                return len(query_value & target_value) / len(query_value)
            return 0.0
            
        elif pattern == "levenshtein":
            return self.levenshtein_distance(query_value, target_value)
        
        return 0.0

    def _compute_distances_vectorized(self, query_row: pd.Series, target_df: pd.DataFrame) -> pd.DataFrame:
        """Compute distances between a query row and all target rows (vectorized where possible)."""
        result_df = pd.DataFrame()
        
        for column, pattern in self.distance_patterns.items():
            query_value = query_row[column]
            column_values: pd.DataFrame = target_df[column]
            
            if pattern == "substraction":
                result_df[column] = np.where(pd.isna(column_values) | pd.isna(query_value), 0, query_value - column_values)
                
            elif pattern == "containment":
                if query_value:
                    result_df[column] = column_values.apply(lambda x: len(query_value & x) / len(query_value) if x else 0.0)
                else:
                    result_df[column] = 0.0
                    
            elif pattern == "levenshtein":
                result_df[column] = column_values.apply(lambda x: self.levenshtein_distance(query_value, x))
        
        # Compute attribute name distances
        result_df["name_dist"] = target_df["attribute_name"].apply(lambda x: self.levenshtein_distance(query_row["attribute_name"], x))
        
        return result_df

    def compute_distances_for_query(self) -> tuple[pd.DataFrame, float]:
        """Generate dataset of distances from the query column to all others."""
        query_row = self.profiles.loc[
            (self.profiles["attribute_name"] == self.query_attribute) 
            & (self.profiles["dataset_name"] == self.query_dataset)
        ].squeeze()
        
        if query_row.empty:
            raise ValueError(f"Query attribute '{self.query_attribute}' in dataset '{self.query_dataset}' not found.")
        
        start_time = time.time()
        
        # Compute distances
        distance_df = self._compute_distances_vectorized(query_row, self.profiles)
        
        # Add metadata columns
        distance_df.insert(0, "dataset_name_2", self.profiles["dataset_name"].values)
        distance_df.insert(0, "attribute_name_2", self.profiles["attribute_name"].values)
        distance_df.insert(0, "dataset_name", self.query_dataset)
        distance_df.insert(0, "attribute_name", self.query_attribute)

        final_time = time.time() - start_time

        safe_name = f"{self.query_dataset.replace('.csv', '_profile')}_{self.query_attribute}".replace("/", "_").replace("\\", "_")
        output_path = self.config.output_distances_path / f"distances_{safe_name}.csv"
        distance_df.to_csv(output_path, index=False)
        
        return final_time

    def compute_distances_for_pair(self, att1: str, ds1: str, att2: str, ds2: str) -> Dict[str, Any]:
        """Generate distances between two specific attribute pairs."""
        row1 = self.profiles.loc[(self.profiles["attribute_name"] == att1) & (self.profiles["dataset_name"] == ds1)]
        row2 = self.profiles.loc[(self.profiles["attribute_name"] == att2) & (self.profiles["dataset_name"] == ds2)]
        
        if row1.empty or row2.empty:
            raise ValueError(f"Could not find profile for ({att1}, {ds1}) or ({att2}, {ds2})")
        
        distances = {
            "attribute_name_1": att1, "dataset_name_1": ds1,
            "attribute_name_2": att2, "dataset_name_2": ds2,
        }

        # Compute distances for each column
        for column, pattern in self.distance_patterns.items():
            distances[column] = self._compute_distance_value(row1[column].iloc[0], row2[column].iloc[0], pattern)
        
        # Compute attribute name distance
        distances["name_dist"] = self.levenshtein_distance(row1["attribute_name"].iloc[0], row2["attribute_name"].iloc[0])
        
        return distances
    
    def generate_distances_for_query(self) -> str:
        """Compute distances for all queries in the benchmark ground truth."""
        error = None
        try:
            elapsed = self.compute_distances_for_query()
        except Exception as e:
            error = f"⚠ Failed for {self.query_attribute} in {self.query_dataset}: {e}"
            logger.error(error)

        logger.info(f"Execution time to compute distances for a single query: {elapsed:.4f} seconds")

        if not error:
            return f"Distances computation fora single query completed (time -> {elapsed:.4f}). Execution finished without errors"
        else:
            return f"Distances computation for a single query failed (time -> {elapsed:.4f}) due to the following error: {error}"

    def generate_distances_for_benchmark(self) -> str:
        """Compute distances for all queries in the benchmark ground truth."""
        queries = self.ground_truth[["target_ds", "target_attr"]].drop_duplicates()

        errors = []
        total_time = 0
        for row in tqdm(queries.itertuples(index=False), total=len(queries)):
            self.query_dataset, self.query_attribute = row.target_ds, row.target_attr
            try:
                elapsed = self.compute_distances_for_query()
                total_time += elapsed

            except Exception as e:
                error = f"⚠ Failed for {self.query_attribute} in {self.query_dataset}: {e}"
                logger.error(error)
                errors.append(error)

        logger.info(f"Execution time to compute distances for benchmark evaluation: {total_time:.4f} seconds")

        if not errors:
            return f"Distances computation for benchmark evaluation completed (time -> {total_time:.4f}). Execution finished without errors"
        else:
            return f"Distances computation for benchmark evaluation completed (time -> {total_time:.4f}). Some tables could not be processed: {errors}"

    def generate_distances_for_training_model(self) -> str:
        """Compute distances for training model pairs from ground truth."""
        results = []
        errors = []
        start_time = time.time()
        
        for row in tqdm(self.ground_truth.itertuples(index=False), total=len(self.ground_truth)):
            try:
                dist_row = self.compute_distances_for_pair(
                    att1=row.att_name, ds1=row.ds_name,
                    att2=row.att_name_2, ds2=row.ds_name_2
                )
                results.append(dist_row)
            except Exception as e:
                error = f"⚠ Failed for {row.att_name} in {row.ds_name} / {row.att_name_2} in {row.ds_name_2}: {e}"
                logger.error(error)
                errors.append(error)

        df_distances = pd.DataFrame(results)
        df_distances.to_csv(self.config.output_distances_path, index=False)

        final_time = time.time() - start_time

        logger.info(f"Done! Execution time: {final_time:.4f} seconds")

        if not errors:
            return f"Distance computation for model training completed (time -> {final_time:.4f}). Execution finished without errors"
        else:
            return f"Distance computation for model training completed (time -> {final_time:.4f}). Some tables could not be processed: {errors}"