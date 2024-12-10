import requests
import pandas as pd

def compute_profile(filePath, storePath):
    url = "http://localhost:8080/profile"

    query_params = {"filePath": filePath, "storePath": storePath}

    # Send the POST request
    response = requests.post(url, params=query_params, verify=False)

    # Check the status code and process the response
    if response.status_code == 200:  # 201 Created
        print(f"Profile created successfully")
        data = response.json()  # Parse JSON response
        return pd.DataFrame(data)
    else:
        print(f"There was a problem when creating the profile")


def compute_profiles_for_folder(directoryPath, storePath):
    url = "http://localhost:8080/profilesOfFolder"

    query_params = {"directoryPath": directoryPath, "storePath": storePath}

    # Send the POST request
    response = requests.post(url, params=query_params, verify=False)

    # Check the status code and process the response
    if response.status_code == 200:  # 201 Created
        print(f"Profiles created successfully")
    else:
        print(f"Failed to send data: {response.status_code}")


def compute_distances_two_files(csvFilePath1, csvFilePath2, pathToWriteDistances):
    url = "http://localhost:8080/computeDistancesTwoFiles"

    query_params = {"csvFilePath1": csvFilePath1, "csvFilePath2": csvFilePath2, "pathToWriteDistances": pathToWriteDistances}

    # Send the POST request
    response = requests.post(url, params=query_params, verify=False)

    # Check the status code and process the response
    if response.status_code == 200:  # 201 Created
        print(f"Distances successfully computed")
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
        print(f"Distances successfully computed")
        data = response.json()  # Parse JSON response
        return pd.DataFrame(data)
    else:
        print(f"Failed to send data: {response.status_code}")


def get_ranking(distancesPath, k, queryDataset, queryAttribute):
    url = "http://localhost:5000/getRanking"

    query_params = {"distancesPath": distancesPath, "k": k, 
                    "queryDataset":queryDataset, "queryAttribute": queryAttribute}

    # Send the POST request
    response = requests.post(url, params=query_params, verify=False)

    # Check the status code and process the response
    if response.status_code == 200:  # 201 Created
        print(f"Ranking computed successfully")
        data = response.json()  # Parse JSON response
        return pd.DataFrame(data)
    else:
        print(f"Failed to send data: {response.status_code}")