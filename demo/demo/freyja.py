import requests
import pandas as pd
import tempfile
import os

data_path = "C:/Projects/FREYJA Demo/data/"

def compute_profile(df):
    url = "http://localhost:8080/profile"
    profiles_path = "C:/Projects/FREYJA Demo/data/profiles"

    # Temporary directory
    system_temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(system_temp_dir, 'd_reference.csv')
    df.to_csv(temp_file_path, index=False)

    query_params = {"filePath": temp_file_path, "storePath": profiles_path}

    # Send the POST request
    response = requests.post(url, params=query_params, verify=False)

    # Check the status code and process the response
    if response.status_code == 200:  # 201 Created
        print(f"Profile created successfully")
        data = response.json()  # Parse JSON response
        return pd.DataFrame(data)
    else:
        print(f"There was a problem when creating the profile")


def compute_profiles_for_folder(directoryPath):
    url = "http://localhost:8080/profilesOfFolder"

    store_path = os.path.join(data_path, "profiles")

    query_params = {"directoryPath": directoryPath, "storePath": store_path}

    # Send the POST request
    response = requests.post(url, params=query_params, verify=False)

    # Check the status code and process the response
    if response.status_code == 200:  # 201 Created
        print(f"Profiles created successfully")
    else:
        print(f"Failed to send data: {response.status_code}")

    # Get all the profiles as dataframe objects and return it
    csv_files = [f for f in os.listdir(store_path) if f.endswith('.csv')]
    dataframes = [pd.read_csv(os.path.join(store_path, file), sep=";") for file in csv_files]

    return dataframes


def compute_distances_two_files(csvFilePath1, csvFilePath2, pathToWriteDistances):
    url = "http://localhost:8080/computeDistancesTwoFiles"

    query_params = {"csvFilePath1": csvFilePath1, "csvFilePath2": csvFilePath2, "pathToWriteDistances": pathToWriteDistances}

    # Send the POST request
    response = requests.post(url, params=query_params, verify=False)

    # Check the status code and process the response
    if response.status_code == 200:  # 201 Created
        # print(f"Distances successfully computed")
        data = response.json()  # Parse JSON response
        return pd.DataFrame(data)
    else:
        print(f"Failed to send data: {response.status_code}")


def compute_distances(queryDataset, queryColumn, profilesFolder, distancesFolder):
    url = "http://localhost:8080/computeDistances"

    query_params = {"queryDataset": queryDataset, "queryColumn": queryColumn, 
                    "profilesFolder": profilesFolder, "distancesFolder": distancesFolder}

    # Send the POST request
    response = requests.post(url, params=query_params, verify=False)

    # Check the status code and process the response
    if response.status_code == 200:  # 201 Created
        # print(f"Distances successfully computed")
        data = response.json()  # Parse JSON response
        return pd.DataFrame(data)
    else:
        print(f"Failed to send data: {response.status_code}")


def get_ranking(query_dataset_profile, query_column, data_lake_profiles, k):

    query_dataset_name = query_dataset_profile["dataset_name"].iloc[0]
    profiles_path = os.path.join(data_path, "profiles")
    distances_path = data_path
    compute_distances(query_dataset_name, query_column, profiles_path, distances_path)
    
    url = "http://localhost:5000/getRanking"

    distances_path = os.path.join(data_path, "distances/")

    query_params = {"distancesPath": distances_path, "k": k, 
                    "queryDataset":query_dataset_name, "queryAttribute": query_column}

    # Send the POST request
    response = requests.post(url, params=query_params, verify=False)

    # Check the status code and process the response
    if response.status_code == 200:  # 201 Created
        print(f"Ranking computed successfully")
        data = response.json()  # Parse JSON response
        ranking = pd.DataFrame(data)
        ranking["normalized_prediction"] = ranking["predictions"].round(5) / 0.37945
        return ranking
    else:
        print(f"Failed to send data: {response.status_code}")

    
def data_augmentation(query_dataset, query_column, ranking, data_path, k):
    row = ranking.iloc[k-1]
    joined_dataset = query_dataset.copy()

    loaded_df = pd.read_csv(os.path.join(data_path, "datalake/" + row['target_ds']))
    joined_dataset = joined_dataset.merge(loaded_df, how='inner', left_on=query_column, right_on=row['target_attr'])

    return joined_dataset