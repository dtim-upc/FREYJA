import joblib
import sklearn 
import pandas as pd
from tqdm import tqdm
import time

def prepare_data_for_model(distances, model):
  distances = distances.drop(columns=['dataset_name', 'dataset_name_2', 'attribute_name', 'attribute_name_2'], axis=1) # Remove unnecesary columns
  distances = distances[model.feature_names_in_] # Arrange the columns as in the model
#   distances = distances.apply(pd.to_numeric, errors='coerce')
  distances = distances.dropna().reset_index(drop=True)
  return distances

def compute_and_evaluate_ranking(model, k, step, ground_truth_path, distances_folder_path):
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

        # Read the distances and do some preprocessing
        distances = pd.read_csv(distances_folder_path + 'distances_' + dataset.replace(".csv", "_profile_") + attribute.replace("/", "_").replace(": ","_").replace("'","_") + ".csv", header = 0, encoding='latin1', on_bad_lines="skip")

        dataset_names = distances["dataset_name_2"] # We store dataset and attribute names to be used to evaluate the ranking
        attribute_names = distances["attribute_name_2"]
        distances = prepare_data_for_model(distances, model)

        # # Use the model to predict
        y_pred = model.predict(distances)
        distances["predictions"] = y_pred

        # Use the model to predict (preventing some weird lines that might have slipped in)
        # distances_numeric = distances.apply(pd.to_numeric, errors='coerce') # Convert everything to float, invalid parsing becomes NaN
        # valid_rows = distances_numeric.dropna(axis=0, how='any') # Keep track of valid rows

        # y_pred = model.predict(valid_rows)

        # distances.loc[valid_rows.index, "predictions"] = y_pred # Assign predictions back only to the valid rows

        distances["target_ds"] = dataset_names
        distances["target_attr"] = attribute_names

        total_time += (time.time() - st) # In the time assessment we do not consider the evaluation of the ranking

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

            top_k_joins = distances.sort_values(by='predictions', ascending=False).head(k_iter * step)

            for position in top_k_joins.itertuples(index=False):
                pair = (position.target_ds, position.target_attr)
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

    print("Precisions:")
    print([round(element / len(pair_counts), 4) for element in precision])

    print("Recall:")
    print([round(element / len(pair_counts), 4) for element in recall])

    print("Max recall:")
    print([round(element / len(pair_counts), 4) for element in max_recall])

    print("Recall percentage:")
    recall_percentage = [round((recall_iter / len(pair_counts)) / (max_recall_iter / len(pair_counts)), 4) for recall_iter, max_recall_iter in zip(recall, max_recall)]
    print(recall_percentage)

    print("MAP:")
    print([round(element / len(pair_counts), 4) for element in MAP])


ground_truth_path = 'path/to/ground_truth.csv'
distances_folder_path = 'path/to/distances/' # Include the final "/"
k = 10
step = 1

model = joblib.load('predictive_model.pkl')
compute_and_evaluate_ranking(model, k, step, ground_truth_path, distances_folder_path)
