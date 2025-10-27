# FREYJA
 
## Introduction
Welcome to FREYJA! This is a tool used to perform data discovery in large-scale environments such as data lakes. The goal of this project is to provide an easy-to-use and lightweight approach for data discovery, that considerably reduces the time requirements of data discovery tasks whilst keeping a high accuracy in the detection of relevant data.

## How to run FREYJA
As of now, FREYJA is an experimental tool, so its full execution is still not correctly orchestrated. More precisely, there are three different processes that need to be performed separately to perform data discovery: creating the profiles, computing distances and executing the predictive model. We will overview each of these steps in the next sections. Bear in mind that these steps need to be repeated for each benchmark that we want to perform data discovery on.

### Requirements
As stated, FREYJA is a lightweight system, which implies that the requirements to execute it ar minimal. Namely, these are:
- Windows system: FREYJA has been developed and tested on Windows, so its correct functionality on other operative systems is not guaranteed.
- Java, preferably Java SDK 21, as it was the one used to develop the tool (can be downloaded [here](https://www.oracle.com/java/technologies/javase/jdk21-archive-downloads.html)).
- Python, preferably Python 3.9, as it was the one used to develop the tool. Alternatively, conda environments can be employed.
- Gradle: you can download and install Gradle from the [official website](https://gradle.org/install/). If on Windows, remember to restart the computer so the changes in the environment variables are stored.

Gradle is required to generate the JAR file that will contain two of the FREYJA functionalities stated above. To do so, clone this repository and navigate to the folder where it is located. Then execute the following commands:

```
gradle build
```
**Note:** In Windows you might need to set the *JAVA_HOME* enviromental variable pointing to the folder of the Java SDK, and then modify the *path* variable so that it also points to *JAVA_HOME/bin*
```
gradle build shadowJar
```

These commands will, respectively, build the project (compile the code, run tests, etc.) and create a [_shadow JAR_](https://github.com/johnrengelman/shadow) (i.e. fat/uber JAR), which contains all classes from dependent jars zipped directly inside it in the correct directory structure. This prevents dependency problems. This JAR can be found in the following directory, under the name of ``FREYJA-all.jar``:
```
cd build\libs\
```

### 1. Generate profiles
Once the JAR has been created, you can generate the profile for a given CSV file in the following manner:

```
java -jar FREYJA-all.jar createProfile path\To\file.csv path\to\store\the\profile
```
#### Automated Script
Generating profiles one by one is a tiresome task. Moreover, the generation of a profile for a given CSV file is independent of the generation of other profiles. This means that we can profit from parallelism to improve performance. To that end, we present a PowerShell script (_generate_profiles.ps1_) that, given a folder, computes the profiles for all the CSV files it contains. By default, it is set to 8 execution threads.

It is recommended to generate the profiles with the script. There are several parameters to define:
- _$directoryBenchmark_: directory of the benchmark to generate the profiles of.
- _$directoryStoreProfiles_: directory to store the resulting profiles to.

Inside the _$block_ script, we find the following line:
```
& "path\to\java.exe" -jar "path\to\FREYJA-all.jar" "createProfile" $file $directoryStoreProfiles "false"
```
- ``"path\to\java.exe"`` indicates the path to the java executable. It is recommended to add the entire path to the Java executable to prevent versioning problems.
- ``"path\to\FREYJA-all.jar"`` indicates the path to the FREYJA jar.
- The `false` at the end indicates whether we want profiles for non-string data. By default, this is set to false as join discovery is tipically done via string-based columns. Nonetheless, non-string data (such as dates) might be used to perform joins. For example, the two OmniMatch benchmarks used in the development of the paper require joins between non-string data, so this parameter has to be set to `true`.

Additionally, the number of execution threads to be used can be modified via the _$MaxThreads_ variable.

After these parameters have been set, the PowerShell can be executed with the following command (in Windows):
```
powershell -File "C:\path\to\generate_profiles.ps1"
```
**Note:** In Windows you might need to change the PowerShell execution policy. To do so, open PowerShell as and administrator and execute the following line:
```
Set-ExecutionPolicy Unrestricted
```

**IMPORTANT NOTE:** Due to an unknown behavior in the DuckDB library (which we employ as a temporary database to generate the profile metrics), each time a profile is generated a temporal file (_libduckdb_java_) is created in the temporal folder, which in Windows is located in:
```
C:\Users\<Username>\AppData\Local\Temp
```
This can quickly consume the space in your drive if not tracked. To prevent so, simply delete these files.

### 2. Computing distances
Once the profiles for a given benchmark have been generated, we can start the data discovery process. Given a _query_column_ from a _query_dataset_, the distances to all other columns in the benchmark can be computed with the following command:
```
java -jar FREYJA-all.jar computeDistances query_dataset.csv query_column path\to\profiles\folder path\to\store\distances
```

This will generate another CSV file, where each row represents the differences between the metrics of the profiles of the _query_column_ and another column in the benchmark. 

#### Automated function
Once again, the task of generating distances one by one is, first, loathsome and, second, easily parallelizable (in this case, without the need for an external script). To facilitate the obtention of distances for a given benchmark we have defined another function in the JAR: _computeDistancesForBenchmark_. **The correct execution of this function requires a ground truth**, as the set of candidate queries will be extracted from it. 

This function can take either three of five additional parameters. This depends on the format of the ground truth used. If the ground truth contains the columns _target_ds_ and _target_attr_ to represent the query dataset and the query attribute, then only three additional parameters are needed. Ground truths from popular benchmarks can be found in the [ground_truths](./ground_truths) folder, with modified headers to fit the abovementioned format. 

```
java -jar FREYJA-all.jar computeDistancesForBenchmark path\to\ground_truth.csv path\to\profiles\folder path\to\store\distances
```

This will generate all distances for all query columns of a benchmark. If the ground truth does not contain _target_ds_ and _target_attr_ as column names, the names of these columns can also be specified:

```
java -jar FREYJA-all.jar computeDistancesForBenchmark path\to\ground_truth.csv header_name_for_query_dataset_column header_name_for_query_attribute_column
```

**Note 1:** By default, the amount of execution threats is set to 8. This can be modified in the ``calculateDistancesForBenchmark`` function, in the [Main](./src/main/java/edu/upc/essi/dtim/FREYJA/Main.java) class.

**Note 2:** By default, the distance are computed between the query column and **all** the columns of the benchmark, which also includes the query column itself. To prevent the computation of distances with itself, go to the [Main](./src/main/java/edu/upc/essi/dtim/FREYJA/Main.java) class and set the last parameter of the ``calculateDistancesAttVsFolder`` function to ``true``.

### 3. Predictive model
The last step is to execute the predictive model, which generates a score for each potential join in the data lake. This represents the predicted quality of the join, with high scores indicating a higher chance of a significant join than lower scores. The set of potential joins can be sorted based on the scores, indicating at the top of the ranking those matches that have the highest potential of being joins.

First, navigate to the [model](./model) folder and install the necessary dependencies using pip:
```
pip install -r requirements.txt
```
It is recommended to use a conda environment to prevent conflicts with the libraries.

The model is contained in the [model](./model) folder (_predictive_model.pkl_). The easiest way to execute the model and obtain a ranking is via the _get_ranking.py_ script. The code requires the following parameters:
- _distance_folder_path_: path to the distances' folder.
- _k_: size of the ranking to be generated.
- _dataset_: name of the query dataset.
- _attribute_: name of the query attribute.
- _model_: path to the model to generate the predictions

Once the parameters are indicated, the script can be run with the following command:
```
python get_ranking.py
```
#### Test performance on benchmarks
To facilitate the computation of evaluation metrics of our system, we have developed another script, which calculates the Precision@k, Recall@k, MaxRecall@k (i.e. maximum possible value of the recall given the characteristics of the benchmark) RecallPercentage@k (i.e. Recall@k / MaxRecall@k) and MAP@k. The code can be found in the _evaluate_benchmark_performance.py_ script.

Similarly as in the _get_ranking.py_ script, it is necessary to introduce the path to the distances folder, the desired k and the model as parameters. However, it is also required to define the path to the ground truth of the benchmark (as it will be used to extract all the query columns) and the _step_, which indicates which are the evaluation points. For instance, given k = 20, we might want to evaluate the scores for every two increments of k (k = 2, k = 4, ..., k = 20). In this case _step_ would be 2.

The script can be executed in the following manner:
```
python evaluate_benchmark_performance.py
```
This will output the abovementioned metrics for the defined benchmark.

## Including new features

To extend the set of features beyond the ones initially defined it is necessary to take the following steps:
- Generate a new function in the [FeatureGeneration.java](./src/main/java/edu/upc/essi/dtim/FREYJA/predictQuality/FeatureGeneration.java) class, with the SQL query to extract the desired metric(s) from the data stored in the temporal DuckDB database. Create a _features_ map and include all metrics obtained from the query.
- Call this newly defined function in the [Profile.java](./src/main/java/edu/upc/essi/dtim/FREYJA/predictQuality/Profile.java) class, including the new features in the _columnFeatures_ variable.
- In the [PredictQuality.java](./src/main/java/edu/upc/essi/dtim/FREYJA/predictQuality/PredictQuality.java) file define is the new metric(s) need(s) to be normalized (if so, add to the list of _metricsToNormalize_) and which can of distance pattern needs to be applied (that is, how do we measure the difference between a given metrics from two profiles). For the latter, most features just take the absolute difference (pattern = 0), although there are also implemented the containment between two sets (pattern = 1), the levenshtein distance between two arrays (pattern = 3) and just including both values from the two profiles (pattern = 2). Define so in the _distancePattern_ variable
- Modify the ending of the _calculateDistances_ function in [PredictQuality.java](./src/main/java/edu/upc/essi/dtim/FREYJA/predictQuality/PredictQuality.java), so that it resets the set of distances if the number of features generated does not match the total (which should be the original 39 + the amount of need features).