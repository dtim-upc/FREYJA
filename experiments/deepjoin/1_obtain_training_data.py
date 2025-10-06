import os
import glob
import random
import duckdb
import pandas as pd
from tqdm import tqdm


def column_comparison_query(table1, col1, table2, col2):
    """Generate SQL query to compare unique value counts and intersections."""
    return f"""
        WITH
            uniq1 AS (
                SELECT DISTINCT "{col1}" AS val FROM "{table1}" WHERE "{col1}" IS NOT NULL
            ),
            uniq2 AS (
                SELECT DISTINCT "{col2}" AS val FROM "{table2}" WHERE "{col2}" IS NOT NULL
            ),
            intersection AS (
                SELECT val FROM uniq1
                INTERSECT
                SELECT val FROM uniq2
            )
        SELECT
            (SELECT COUNT(*) FROM uniq1) AS unique_count_col1,
            (SELECT COUNT(*) FROM uniq2) AS unique_count_col2,
            (SELECT COUNT(*) FROM intersection) AS intersection_count;
    """


def load_csv_files(con, folder_path, exclude_keywords=None):
    """Load all CSV files into DuckDB tables, excluding files with a keyword."""
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if exclude_keywords:
        if isinstance(exclude_keywords, str):
            exclude_keywords = [exclude_keywords]
        csv_files = [
            f for f in csv_files 
            if not any(keyword in f for keyword in exclude_keywords)
        ]

    for file in tqdm(csv_files, desc="Materializing tables"):
        table_name = os.path.splitext(os.path.basename(file))[0]
        con.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        con.execute(f'''
            CREATE TABLE "{table_name}" AS
            SELECT * FROM read_csv_auto("{file}", IGNORE_ERRORS=TRUE, STRICT_MODE=false)
        ''')

    return csv_files


def get_all_columns(con, csv_files):
    """Retrieve all (table, column) pairs."""
    all_columns = []
    for file in tqdm(csv_files, desc="Getting all columns"):
        table_name = os.path.splitext(os.path.basename(file))[0]
        cols = [c[0] for c in con.execute(f'DESCRIBE "{table_name}"').fetchall()]
        all_columns.extend((table_name, col) for col in cols)
    return all_columns


def generate_valid_pairs(all_columns):
    """
    Lazily yield valid column pairs between different tables in random order.
    Avoids materializing the full O(n^2) pair list.
    """
    n = len(all_columns)
    indices = list(range(n))
    random.shuffle(indices)

    for i in indices:
        t1, c1 = all_columns[i]
        for j in indices:
            if i < j:
                t2, c2 = all_columns[j]
                if t1 != t2:
                    yield (t1, c1, t2, c2)


def find_positive_pairs(con, all_columns, max_pairs, containment_threshold):
    """
    Find up to `max_pairs` positive column pairs using lazy pair generation.
    Stops early once enough positives are found.
    """
    results = []
    positive_count = 0
    total_checked = 0

    progress = tqdm(total=max_pairs, desc="Positive joins found", position=1)

    for table1, col1, table2, col2 in generate_valid_pairs(all_columns):
        total_checked += 1

        try:
            count_a, count_b, intersection = con.execute(
                column_comparison_query(table1, col1, table2, col2)
            ).fetchone()
        except (duckdb.ConversionException, duckdb.BinderException, duckdb.ParserException):
            continue

        if count_a > 0 and count_b > 0:
            containment = intersection / min(count_a, count_b)
            if containment >= containment_threshold:
                results.append({
                    "table1": table1,
                    "column1": col1,
                    "table2": table2,
                    "column2": col2,
                    "containment": containment,
                })
                positive_count += 1
                progress.update(1)

                if positive_count >= max_pairs:
                    break

    progress.close()
    print(f"Found {positive_count} qualifying pairs after checking {total_checked} pairs.")
    return pd.DataFrame(results)


def create_negative_examples(con, positives_df, negatives_per_positive=3):
    """Generate negative column pairs for training."""
    positives = positives_df.to_dict("records")
    negative_examples = []

    for pos in tqdm(positives, desc="Creating negative examples"):
        for _ in range(negatives_per_positive):
            while True:
                other = random.choice(positives)
                if not (
                    other["table1"] == pos["table1"]
                    and other["column1"] == pos["column1"]
                    and other["table2"] == pos["table2"]
                    and other["column2"] == pos["column2"]
                ):
                    break

            try:
                count_a, count_b, intersection = con.execute(
                    column_comparison_query(pos["table1"], pos["column1"], other["table2"], other["column2"])
                ).fetchone()
            except (duckdb.ConversionException, duckdb.BinderException, duckdb.ParserException):
                continue

            containment = intersection / min(count_a, count_b) if (count_a > 0 and count_b > 0) else 0
            negative_examples.append({
                "table1": pos["table1"],
                "column1": pos["column1"],
                "table2": other["table2"],
                "column2": other["column2"],
                "containment": containment,
            })

    return pd.DataFrame(negative_examples)

benchmark_name = "omnimatch_culture_recreation"
benchmark_folder = f"../benchmarks/{benchmark_name}/datalake"
max_positive_pairs = 5000
containment_threshold = 0.7

con = duckdb.connect(database=':memory:')

# Load CSV files and register them as tables
csv_files = load_csv_files(
    con,
    benchmark_folder,
    exclude_keywords=["t_013a2f8c584d44d7", "t_70941cace7dd1c45"] # Problematic files from TUS
)

all_columns = get_all_columns(con, csv_files) # Get all columns from all tables
positive_examples = find_positive_pairs(con, all_columns, max_positive_pairs, containment_threshold) # Find positive matches lazily
negative_examples = create_negative_examples(con, positive_examples) # Create negative matches (operates only on positives, safe size)

# Combine and save results
training_data = pd.concat([positive_examples, negative_examples], ignore_index=True)
data_path = f"data/{benchmark_name}"
os.makedirs(data_path, exist_ok=True)
training_data.to_csv(os.path.join(data_path, f"raw_training_joins.csv"), index=False)
