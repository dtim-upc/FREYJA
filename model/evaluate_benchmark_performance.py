import joblib
import sklearn 
import pandas as pd
from tqdm import tqdm
import time

def prepare_data_for_model(distances, model):
  distances = distances.drop(columns=['dataset_name', 'dataset_name_2', 'attribute_name', 'attribute_name_2'], axis=1) # Remove unnecesary columns
  distances = distances[model.feature_names_in_] # Arrange the columns as in the model
  distances = distances.dropna()
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
      distances = pd.read_csv(distances_folder_path + 'distances_' + dataset.replace(".csv", "_profile_") + attribute.replace("/", "_").replace(": ","_") + ".csv", header = 0, encoding='latin1', on_bad_lines="skip")

      dataset_names = distances["dataset_name_2"] # We store dataset and attribute names to be used to evaluate the ranking
      attribute_names = distances["attribute_name_2"]
      distances = prepare_data_for_model(distances, model)

      # Use the model to predict
      y_pred = model.predict(distances)
      distances["predictions"] = y_pred

      distances["target_ds"] = dataset_names
      distances["target_attr"] = attribute_names

      total_time += (time.time() - st) # In the time assessment we do not consider the evaluation of the ranking

      # For every k that we want to assess the ranking of, we get the top k joins and check how many appear in the grpund truth
      for k_iter in range(0, num_observations):
        top_k_joins = distances.sort_values(by='predictions', ascending=False).head((k_iter + 1) * step)

        count_sem = 0
        ap = 0
        for i in range(0, (k_iter + 1) * step):
            top_k_join = top_k_joins.iloc[i]
            result = ground_truth[(ground_truth['target_ds'] == dataset) & (ground_truth['target_attr'] == attribute) &
                                        (ground_truth['candidate_ds'] == top_k_join["target_ds"]) & (ground_truth['candidate_attr'] == top_k_join["target_attr"])]
            if not result.empty:
                count_sem = count_sem + 1
                ap = ap + (count_sem/(i + 1))
        precision[k_iter] = precision[k_iter] + (count_sem/((k_iter + 1) * step))
        if (count_sem != 0):
          MAP[k_iter] = MAP[k_iter] + (ap/count_sem)
        recall[k_iter] = recall[k_iter] + (count_sem/count)
        max_recall[k_iter] = max_recall[k_iter] + (((k_iter + 1) * step)/count)

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
k = 20
step = 2

model = joblib.load('predictive_model.pkl')
compute_and_evaluate_ranking(model, k, step, ground_truth_path, distances_folder_path)
