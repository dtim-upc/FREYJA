# FREYJA
**Important note:** FREYJA was originally built using Java. The initial Java implementation can be fond under the *java_code* branch. Nonetheless, we highly recommend using the current, Python-based implementation (*main* branch), as it is less prone to errors and considerably faster.
 
## Introduction
Welcome to FREYJA! This is a tool used to perform data discovery in large-scale environments such as data lakes. The goal of this project is to provide an easy-to-use and lightweight approach for data discovery, that considerably reduces the time requirements of data discovery tasks whilst keeping a high accuracy in the detection of relevant data.

We focus on the task of **join discovery** for **tabular data**: given a datalake (i.e. a folder with CSV files) and a query column (i.e. the name of a column of one of the CSV files in the datalake, for which to perform the discovery), the goal is to find other columns (candidate columns) in the datalake that can perform joins with the query column, thus combining the information of several datasets into a single file, increasing the amount of information for downstream analyses. FREYJA ranks all the candidate columns from best to worst, according to the quality of the join.

For further details on how FREYJA works, please refer to the original paper - [FREYJA: Efficient Join Discovery in Data Lakes](https://arxiv.org/abs/2412.06637). This paper was accepted for publication in TKDE in January 2026. Since then, we have updated the code and performed efficiency improvements. The previous link points to the most up-to-date version of the paper, while also being open access.

Other details, benchmark information and results analyses can be found in the [accompanying website](https://freyja-data-discovery.github.io/).

## Setting up Freyja
As of now, FREYJA is an experimental tool, so its full execution is still not fully orchestrated. In any case, operating with Freyja should be quite straightforward, as we offer a simple API to interact with all the necessary functions.

More precisely, there are three different processes that need to be performed separately to perform data discovery: (i) creating data profiles, (ii) computing distances and (iii) executing the predictive model. We will overview each of these steps in the next sections. Step (i) corresponds to the offline phase (i.e. preprocessing), which needs to be executed only once per datalake. Steps (ii) and (iii) encompass the online phase (i.e. the discovery process), and need to be repeated everytime that we want to perform a discovery task over a given column

### Requirements
As stated, FREYJA is a lightweight system. It is built entirely in Python (so there is no need to employ other software) and does not require the installation of hefty libraries.

The only strong requirement is Python, preferably version 3.9. We recommend using [conda](https://www.anaconda.com/docs/getting-started/miniconda/install#windows-installation) to create environments for an easy handling of the dependencies. Once conda is installed, you can open a CLI and execute the following command to create an environment:

```
conda create --name freyja python=3.9
```
This will create an environment named *freyja* with base python 3.9. Then, you can activate the environment with:
```
conda activate freyja
```
Then, we just need to install the dependencies. To do so, navigate to the root folder of the Freyja repository and execute the following command:
```
pip install -r requirements.txt
```

### Deploying Freyja
We recommend deploying Freyja's API to facilitate the execution of the discovery process. To do so, navigate to the root folder of the Freyja repository and execute the following command (with the conda environment activated):
```
uvicorn app.main:app --reload
```
This should deploy the API on localhost:8000 (you should see a message stating *Uvicorn running on http://127.0.0.1:8000*). Also, if you open a browser window and navigate to *http://localhost:8000/docs#/*, you should see the Swagger specification of the available API requests. You can interact with Freyja simply by filling the parameters of the requests for each functon (after clicking *try it out*). Alternatively, API-development tools such as Postman or raw CURL requests can be use.

Alternatively, Freyja's functions can be executed directly from the scripts, without deploying the API. These can be found in [app/core](./app/core).

## Using Freyja
### 1. Generate profiles
The first step in the Freyja pipeline is to obtain the profiles of all the datasets in the datalake. A profile is simply a set of descriptive statistics (e.g. entropy, incompletness, average length of strings, maximum frequency of a value, etc.) computed for every column. This is the only preprocessing that we need, as from this point onwards the profiles are the only artifact that will be used to perform the discovery process.

Request endpoint -> POST http://localhost:8000/profiles/profile_datalake

There are four parameters that can be altered here:
- `datalake_path` (string): this needs to point to the folder where the CSV files are located.
- `output_profiles_path` (string): path where the profiles will be stored. Note that all the profiles will be stored in a single file, so this path has to end in something like `/profiles_benchmark.csv`
- `varchar_only` (boolean): flag that indicates whether numerical columns should be profiled or not. Recall that we are performing join discovery, a task mainly associated with string-based columns given the incresed difficulty of discovering joins for numerical columns. Hence, this parameter is set to `true` by default. Nonetheless, in some cases it might be necessary to find joins for numerical data, so turning this flag to `false` allows to do so. Note that this will slow down the process and the final results might be compromised, both because there is more data to work with.
- `max_workers` (int): number of parallel executions. To speed up the process of computing profiles, Freyja parallelizes the execution. For maximum efficiency, this should be equal to the amount of cores of your computer, but we allow the user to input any value to tune how much resources are devoted to Freyja. The default value is 8.

If our datalake is located in `C:/Projects/freyja/test_benchmark`, an example of the request's body could be:
```
{
  "datalake_path": "C:/Projects/freyja/my_benchmark",
  "output_profiles_path": "C:/Projects/freyja/profiles/profiles_my_benchmark.csv",
  "varchar_only": true,
  "max_workers": 8
}
```
In the CLI where Freyja was deployed, you should see a progress bar indicating the amount of files left to be processed. Once it is finished, the total time required to generate the profiles will be printed in the CLI and returned as the request response. If there were any errors with conflicting files, it will also be notified.

This will create three files in `C:/Projects/freyja/profiles`, all containing one row per column in the datalake, with the cell values corresponding to each of the defined profile metrics.
- *profiles_my_benchmark.csv*: raw metrics obtained from the profiling.
- *profiles_my_benchmark_**normalized**.csv*: profiling metrics after being normalized. That is, we apply Z-score normalization to facilitate further processing of the data.
- *profiles_my_benchmark_normalized.**pkl***: exact same content as *profiles_my_benchmark_**normalized**.csv*, but with a preprocessing applied. This preprocessing simply implies creating a pandas dataframe and assigning the correct typologies to every column. Doing so one time prevents the need of having to adapt the data repeatedly in subsequent stages.

The **only file** we will be employing in the next steps is *profiles_my_benchmark_normalized.**pkl***. We generate the other two files (the CSVs) to allow the user to inspect of the profiles, both the raw values as well as after the normalization.

### 2. Computing distances
Once the profiles for a given benchmark have been generated, we can start the data discovery process. The first step is to compute the profiles distances. That is, given a query column, we compute the distance (i.e. difference) of every of the metrics of its profile w.r.t. to the profiles of all the other columns in the datalake (pair-wise). This will provide an approximated notion of how much each column differs from structural properties from the query columns. That is, if the differences between the profile of the query column and the profile of a candidate column are very high, it is likely that the candidate will not produce a relevant join.

There are three functions here. The two main functions to be executed for most users are `distances_for_query` and `distances_for_benchmark`. The former executes a single discovery process, whereas the latter executes several discovery processes sequentially, and it is useful to test data discovery benchmarks.

#### 2.1 Computing distances for queries
Request endpoint -> POST http://localhost:8000/distances/distances_for_query

There are four parameters that can be altered here:
- `profiles_file_path` (string): path to where the profiles to be employed are located. This should point to the .pkl file created beforehand.
- `output_distances_path` (string): path where the distances will be stored.
- `query_column` (string): name of the column to perfom join discovery for.
- `query_dataset` (string): name of the datase that contains the query column. This has to end in .csv.

Following the previous example, we could perform the following discovery task:
```
{
  "profiles_file_path": "C:/Projects/freyja/profiles/profiles_my_benchmark_normalized.pkl",
  "output_distances_path": "C:/Projects/freyja/distances/my_benchmark",
  "query_column": "country_name",
  "query_dataset": "countries_data.csv"
}
```
This process should be quite fast, as we are employing vectorized computation for the metrics' distances and we are opèrating over an already preprocessed dataframe. The output will be a single CSV file: *distances_<query_dataset>_<query_column>.csv*, with one row for every candidate column in the datalake.

#### 2.2 Computing distances for benchmarks
Request endpoint -> POST http://localhost:8000/distances/distances_for_benchmark

There are three parameters that can be altered here:
- `profiles_file_path` (string): path to where the profiles to be employed are located. This should point to the .pkl file created beforehand.
- `output_distances_path` (string): path where the distances will be stored.
- `ground_truth_path` (string): path where the ground truth of the benchmark is located.

The ground truth will be used to automatically obtain the query columns and query datasets, without the user having to specify them. These files contains, for a given query column, all the candidate columns that *should* ideally be found, allowing to test how accurate the system is.

Examples of ground truths can be found in the [ground_truths folder](./ground_truths). To obtain the distances, the only columns used are `target_ds` and `target_attr` (i.e. the query dataset and the query column). It is important that any ground truth contains, at least, these two columns. Note that we only take unique combinations of `target_ds` and `target_attr`, as we are interested in knowing which columns we have to discover joins for, rather than assessing how correct they are.

If we wanted to compute joins for the `santos_small` benchmark, we could fill the parameters in the following manner (assuming that we have already obtained the profiles):
```
{
  "profiles_file_path": "C:/Projects/freyja/profiles/profiles_santos_small_normalized.pkl",
  "output_distances_path": "C:/Projects/freyja/distances/santos_small",
  "ground_truth_path": "C:/Projects/freyja/ground_truths/santos_small_ground_truth.csv"
}
```
Similarly as with the profile computation, a progress bar will appear in CLI. Both the total and average time to compute the distances will be displayed. Note that this time does not consider the time required to write the distances into the CSV files, as it is not strictly needed (i.e. we could simply send the distances to the final step, without having to store the intermediate results).

The output now will be one CSV file for every combination of `query_dataset` and `query_column`.

### 3. Obtaning join quality rankings (via a predictive model)
The final step of Freyja consists on executing a predictive model to obtain a ranking of the candidate columns. This model has been trained with profiles' differences with the goal of infering a measure of join quality (for more information, we refer to the Freyja paper at the beggining of this README). Now, we will employ this pre-trained model to infer this quality metric for new pairs of columns. The candidate columns will be ranked according to this inferred quality, with those columns at the top being the most relevant to perform joins with.

Similarly as in the previous step, there are two functions to consider here, depending on whether we want to perform a single discovery process or test and entire benchmark: `obtain_ranking` and `evaluate_benchmark`. The former obtains a single ranking given a query column, whereas the latter evaluates the join process of an entire benchmark.

#### 3.1 Obtaining rankings for queries
Request endpoint -> POST http://localhost:8000/model/obtain_ranking

There are three parameters that can be altered here:
- `model_path` (string): path where the model that will process the distances is located. We have obtained this model by pre-training it with a large amount of distances, and this variable does not need to be changed unless a new model has been trained. By deafult, this model is located [here](./app/core/model).
- `distances_path` (string): path where the distances are located (the CSV file). Note that the distances' file name already include the names of the query column and query dataset, so it is not necessary to insert them again.
- `top_k` (integer): number of elements to display in the quality ranking. That is, how many of the top candidates joins should be displayed.

Following the example of section 2.1, we could have the following parameters:
```
{
  "model_path": "C:/Projects/freyja/app/core/model/gradient_boosting_ne100_lr0.05_md3_ss0.8_msl10.pkl",
  "distances_path": "C:/Projects/freyja/distances/my_benchmark/distances_country_name_country_data.csv",
  "top_k": 10
}
```
The result will be a display of the top k elements, sorted by their score, and the time needed to execute the model. Two relevant points:
- The score itself is not meaningful, what is relevant is the ordering.
- The query column will appear in the first position of the ranking.

#### 3.2 Obtaining rankings for benchmarks
Request endpoint -> POST http://localhost:8000/model/evaluate_benchmark

There are five parameters that can be altered here:
- `k`(integer): number of elements to display in the quality ranking. Join discovery systems tend to evaluate the join detection capacity with varying number of elements. For example, take the top 5 elements and evaluate the join detection precision, then the top 10, then the top 15, and so on until 50. This `k` should be the maximum size of the ranking (in the example, it should be 50).
- `step` (integer): this indicates how much we jump at every iteration of the evaluation process. Following the previous example, `step` should be 5. The `k`and `step` values for the testes benchmarks are indicated [here](./ground_truths/README.md)
- `ground_truth_path` (string): path where the ground truth of the benchmark is located.
- `distances_folder_path` (string): path where the distances are located. Note that as now we are evaluating the benchmark rather than a specific query, this path has to point to the folder where all the distances corresponding to the benchmark are located (i.e. it should coincide with the `output_distances_path` of 2.2).
- `model_path` (string): path where the model that will process the distances is located. We have obtained this model by pre-training it with a large amount of distances, and this variable does not need to be changed unless a new model has been trained. By deafult, this model is located [here](./app/core/model).

Following the example of section 2.2, we could have the following parameters:
```
{
  "k": 10,
  "step": 1,
  "ground_truth_path": "C:/Projects/freyja/ground_truths/santos_small_ground_truth.csv",
  "distances_folder_path": "C:/Projects/freyja/distances/santos_small",
  "model_path": "C:/Projects/freyja/app/core/model/gradient_boosting_ne100_lr0.05_md3_ss0.8_msl10.pkl"
}
```
In this case the result, rather than being the specific datasets and columns that appear on the top positions of the ranking, is composed of averaged metrics. First, we display the precision, recall and MAP for the benchmark. Note that the maximum recall can not always be 1 (due to the characteristics of the benchmarks), so we compute the "normalized" recall to provide a fair comparison. These three measurements are obtained for every "step" level. For example, if k is 10 and step is 1, there have been 10 evaluations of the ranking, with varying amounts of joins selected. Therefore, each of the measurements will contain 10 components. Finally, each of the components is the average for all the query columns. For example, for step = 1 (i.e. we only take the top-most quality join into account) we compute the precision, recall and MAP for all the query columns, and we average their results to obtain the final value.

We also display the total and average time to compute the rankings. This time alongside the time to obtain the distances results in the total execution time for the online phase.