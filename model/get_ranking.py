import joblib
import sklearn 
import pandas as pd

def prepare_data_for_model(distances, model):
  distances = distances.drop(columns=['dataset_name', 'dataset_name_2', 'attribute_name', 'attribute_name_2'], axis=1) # Remove unnecesary columns
  distances = distances[model.feature_names_in_] # Arrange the columns as in the model
  distances = distances.dropna()
  return distances


def get_ranking(distances_folder_path, k, dataset, attribute, model):
  # Read distances
  distances = pd.read_csv(distances_folder_path + 'distances_' + dataset.replace(".csv", "_profile_") + attribute.replace("/", "_").replace(": ","_") + ".csv", header = 0, encoding='latin1', on_bad_lines="skip")

  dataset_names = distances["dataset_name_2"] # We store dataset and attribute names to be used to evaluate the ranking
  attribute_names = distances["attribute_name_2"]
  distances = prepare_data_for_model(distances, model) # Prepare the data

  y_pred = model.predict(distances) # Use the model to predict
  distances["predictions"] = y_pred

  distances["target_ds"] = dataset_names
  distances["target_attr"] = attribute_names

  top_k_joins = distances.sort_values(by='predictions', ascending=False).head(k)
  print(top_k_joins.head(k))

distances_folder_path = 'path/to/distances/' # Include the final "/"
k = 20
dataset = "dataset.csv"
attribute = "attribute"
model = joblib.load('predictive_model.pkl')

get_ranking(distances_folder_path, k, dataset, attribute, model)