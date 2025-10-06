import os
import pandas as pd
import random
from tqdm import tqdm

def load_datasets(folder_path):
    """Load all CSV files in a folder into a dictionary: dataset_name -> DataFrame"""
    datasets = {}
    for filename in tqdm(os.listdir(folder_path), desc="Loading datasets"):
        if filename.endswith(".csv"):
            dataset_name = os.path.splitext(filename)[0]
            df = pd.read_csv(os.path.join(folder_path, filename), encoding="latin1")
            datasets[dataset_name] = df
    return datasets

def column_to_text(column, title=None):
    """
    Generate descriptive column text:
    Title. colname contains N values (max, min, avg): val1, val2, ...
    """
    colname = column.name
    cells = column.dropna().astype(str)
    n_unique = len(cells.unique())
    word_counts = cells.apply(lambda x: len(x.split()))
    if len(word_counts) == 0:
        max_len = min_len = avg_len = 0
    else:
        max_len = word_counts.max()
        min_len = word_counts.min()
        avg_len = round(word_counts.mean(), 2)
    # get unique values
    unique_vals = cells.unique()

    # set a limit for how many to show
    max_display = 2000  

    if len(unique_vals) > max_display:
        # take a random sample
        sampled_vals = random.sample(list(unique_vals), max_display)
        unique_vals_text = ", ".join(map(str, sampled_vals)) + f" ... (+{len(unique_vals) - max_display} more)"
    else:
        unique_vals_text = ", ".join(map(str, unique_vals))
    text = f"{title}. {colname} contains {n_unique} values ({max_len}, {min_len}, {avg_len}): {unique_vals_text}."
    return text


def generate_training_examples(datasets, raw_training_joins):
    """Generate training examples with positives and negatives"""
    training_joins = []
    for _, row in tqdm(raw_training_joins.iterrows(), desc="Creating text from joins", total=len(raw_training_joins)):
        try:
            table1_name = row["table1"]
            col1_name = row["column1"]
            table2_name = row["table2"]
            col2_name = row["column2"]
            containment = row["containment"]

            table1 = datasets[table1_name]
            col1 = table1[col1_name]
            table2 = datasets[table2_name]
            col2 = table2[col2_name]

            col_text1 = column_to_text(col1, title=f"{table1_name} information")
            col_text2 = column_to_text(col2, title=f"{table2_name} information")

            training_joins.append([col_text1, col_text2, containment])
        except:
            continue

    return pd.DataFrame(training_joins, columns=["col_text1", "col_text2", "containment"])


if __name__ == "__main__":
    benchmark_name = "omnimatch_culture_recreation"
    benchmark_folder = f"../benchmarks/{benchmark_name}/datalake"
    raw_training_joins = pd.read_csv(f"data/{benchmark_name}/raw_training_joins.csv")

    datasets = load_datasets(benchmark_folder)
    train_df = generate_training_examples(datasets, raw_training_joins)

    train_df.to_csv(f"data/{benchmark_name}/train_data.csv", index=False)
