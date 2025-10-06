# Deepjoin
This repository contains our own implementation for [*DeepJoin: Joinable Table Discovery with Pre-trained Language Models*](https://arxiv.org/abs/2212.07588). This work introduces a deep learning framework designed to efficiently and accurately identify joinable tables within data lakes. Unlike traditional methods that rely on exact matches or approximate solutions with limited precision, DeepJoin employs a pre-trained language model (PLM) to embed column contents into a fixed-length vector space. By fine-tuning the PLM, the system ensures that semantically similar or equi-joinable columns are positioned close together in this vector space. 

This approach supports both equi-joins and semantic joins, accommodating variations in spelling and formatting. To enhance retrieval efficiency, DeepJoin utilizes a state-of-the-art approximate nearest neighbor search algorithm, achieving sublinear search time relative to the repository size. The model is trained using a self-supervised method, generating positive and negative examples through data augmentation techniques. Experimental results demonstrate that DeepJoin outperforms existing solutions in both precision and scalability, with performance improvements of up to two orders of magnitude when equipped with a GPU.

DeepJoin acts as a relevant baseline for our analyses, given that it presents the natural evolution of initial embedding-based approaches: rather than taking an embedding model and use it as-is, fine-tune it to better perform in the specific scenario of join detection in data lakes. Unfortunately, no open implementation is available. Hence, we developed our own version following the indications of the original paper. Additionally, we contacted the original authors, which provided useful guidelines and recommended parameters configurations.

## Implementation Details
DeepJoin consists on 5 major steps:

1. **Obtain training data**. Given that we have to fine-tune the embeddings, we need the necessary training data to do so. This training data consists of pairs of columns that can be classified into two main groups: (i) positive examples, that is, pairs of columns with “high” joinability and (ii) negative examples, that is, pairs of joins with “low” joinability. Following the recommendations of the authors, we assessed the join quality based on containment, with positive examples being those pairs of columns with a containment > 0.7. The negative examples were obtained using  in-batch negatives, that is, shuffling the pairs identified as positive examples to create uncorrelated joins. To thoroughly train the model, we obtain 5.000 positive examples and generate 15.000 negative examples, for each benchmark.

Note: This is the main bottleneck of DeepJoin, as it requires searching for “good” joins in each datalake, which implies that the larger and the more heterogeneous that a data lake is, the more time will be required to find these positive examples, increasing exponentially. The execution times stated for DeepJoin follow the previously stated configuration except for the TUS Big and Santos Big benchmarks due to the excessive amount of time required to find all of these joins. For the TUS Big dataset we collected just 1.000 positive examples and for Santos Big this was reduced to 100 positive examples. Otherwise, the time becomes prohibitive.

2. **Prepare the training data**. The model that will generate the embeddings is a standard SBERT, which requires text. Hence, following the recommendations of the paper, for each column of the benchmark we generate its textual representation by concatenating: table name, column name: number of distinct values, maximum, minimum and average number of characters of the strings in the columns and the list of values themselves (*title-colname-stat-col*, as defined in the paper).

3. **Train the bi-encoder model**. As stated, we will fine-tune the embeddings to perform join detection. Following the authors’ recommendations we employ SBERT as base model. More precisely, we use the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model (384-dimensional embeddings) with [mnrloss](https://huggingface.co/blog/dragonkue/mitigating-false-negatives-in-retriever-training) as the loss function, which only works on binary labels. Hence, all positive examples are assigned the label “1” and all negative examples the label “0”. The original paper indicates the following values for the model’s hyperparameters, which we also adopt: `batch size = 32`, `learning rate = 2e-5`, `warmup steps = 10.000`, and `weight decay rate = 0.01`. As for the number of epochs, we tried to balance training time with accuracy, so we followed a dynamic approach that reduced the number of epochs from 20 to 5 as the size of the tables increased (from less than 5.000 to over 1.000.000).

4. **Generate embeddings**. After training, we generate the embeddings of all the columns in each benchmark. Batch size is fixed at 256.

5. **Store embeddings in index**. Similarly as with WarpGate, we store the embeddings in an index to perform quick retrieval and comparison. Here, though, instead of employing LSH index, the authors recommend to use a Hierarchical Navigable Small World (HNSW) index through the FAISS Python library that combines HNSW with IVF, PQ and GPU acceleration. We employ the recommended configuration: `hnsw_m = 32`, `hnsw_ef_search = 64`, `nlist = 1000`, `pq_m = 16` and `pq_bits = 8`. The similarity assessment is done via cosine similarity, so we normalize the embeddings before storing them. At query time, the index is traversed to retrieve the k nearest neighbors.

## Execution Instructions

### Requirements
The only requirement to execute is Python and a set of libraries. The recommended Python version is 3.9 and the set of libraries is specified in the `requirements.txt` file. You can install them with the following command:
```
pip install -r requirements.txt
```
To prevent dependency issues, it is recommended to create a virtual environment using conda.

## Step 1: Obtaining Training Data
The first step is to find the positive and negative examples to train the model. The script `1_obtain_training_data.py` contains the code to do so. This script generates a training dataset for column matching by loading multiple CSV files (i.e. the datalake) into an in-memory DuckDB database, then identifying pairs of columns from different tables that share a significant fraction of values (positive examples) and pairing other columns with low or no overlap (negative examples), that is, in-batch negatives. It computes a containment metric for each pair based on the intersection of unique values, efficiently samples column pairs to avoid O(n²) computations.

The result will be a combined CSV file of positive and negative column pairs with their containment scores (`raw_training_joins`). By default, this will placed in a subfolder inside the repository (`data/{benchmark_name}`).

The following parameters need to be filled:
- `benchmark_name`: name of the datalake. This is used to define the name where the resulting data will be stored. 
- `benchmark_folder`: path to where the datalake is located
- `max_positive_pairs`: amount of positive pairs to find (5,000 by default). The more, the better the learning performed by the model, although higher values might incur higher time costs.
- `containment_threshold`: containment value that a join needs to surpass in order to be considered a positive join (0.7 by default)

Once the set up is done, the code can be executed via the following command:
```
python 1_obtain_training_data.py
```
## Step 2: Prepare Training Data
Once we have our raw training pairs, we will transform them into text so they can be ingested by the base model (BERT). The code in `2_prepare_training_data.py` converts CSV datasets and previously computed column pairs into a text-based training dataset by loading all CSV files into pandas DataFrames, generating descriptive text for each column (including the number of unique values, word length statistics, and a sample of values) and pairing these textual representations according to the positive and negative matches in a raw training joins file

The resul is a CSV where each row contains the text of two columns and their containment score for use in machine learning models. This file is, by default, placed in the same folder as `raw_training_joins` (i.e. in `data/{benchmark_name}`).

The following parameters need to be filled:
- `benchmark_name`: name of the datalake. This is used to define the name where the resulting data will be stored. 
- `benchmark_folder`: path to where the datalake is located

If you have moved the data outside of the repository folder, you will need to define the path where the `raw_training_joins` are.

Once the set up is done, the code can be executed via the following command:
```
python 2_prepare_training_data.py
```

## Step 3: Train Bi-Encoder Model
With our data ready, `3_train_bi_encoder.py` trains a SentenceTransformer model to embed column text pairs by loading previously he generated text-based training pairs with containment scores above a threshold, converting them into InputExamples for a DataLoader, determining suitable training parameters (epochs and warmup steps) based on dataset size, and fine-tuning a pre-trained transformer using the MultipleNegativesRankingLoss.

The resulting model is saved (by default, in `models/` folder).

The following parameters need to be filled:
- `benchmark_name` = name of the datalake. This is used to define the name where the resulting data will be stored.
- `positive_threshold` = threshold that will be used to differentiate positive cases (labelled 1 for the loss function) from negative cases (labelled 0). By default, it is 0.7, in accordance to the value used in the first script.
- `base_model` = hugging face name for the model to be used. to generate the embeddings. By default, `sentence-transformers/all-MiniLM-L6-v2`

If you have moved the data outside of the repository folder, you will need to define the path of the training data (`train_path`) and the path where the model is stored (`output_dir`).

Once the set up is done, the code can be executed via the following command:
```
python 3_train_bi_encoder.py
```

## Step 4: Embed Columns
We will now employ the generated model to produce embeddings of all the columns in the datalake. `4_embed_columns` generates vector embeddings for all columns in a set of CSV files by converting each column into descriptive text (same as `2_prepare_training_data` but for the entire data lake), encoding these texts using a the fine-tuned model.

The result is a ready-to-use representation of all columns, alongside metadata for later indexing (both in CSV and JSON format). Note that every column is identified with a unique number to be later retrieved.

The following parameters need to be filled:
- `benchmark_name` = name of the datalake. This is used to define the name where the resulting data will be stored.

If you have moved the data outside of the repository folder, you will need to define the path where the datalake resides (`benchmark_folder`), the path where the model has been saved (`model_path`) and the path where the embeddings will be stored (`output_dir`, by default in the `embeddings` subfolder)

Once the set up is done, the code can be executed via the following command:
```
python 4_embed_columns.py
```

## Step 5: Semantic Search
Once the embeddings have been obtained, we can perform similarity search to detect similar columns; file `2_search.py`.

First, we create the HNSW index (optionally, we can build a HNSW + IVFPQ index, recommended for very large benchmarks). Note that you can modify the creation parameters, although the default values are the recommended ones.

You will also find the `compute_and_evaluate_ranking` function. This function is employed to obtain precision, recall (raw, maximum possible and percentual) and MAP scores, as those observed in the paper. If you do not want to use this function, you can simply perform queries over the index.

In both cases, you will need to define the following parameter:
- `benchmark_name`: assuming that you are using the default directory for the data generated in the previous script, this name is used to automatically obtain the embeddings and metadata needed.
- `mode`: type of index desired (`"hnsw"` or ``"ivfpq"``).

Optionally, you can modify the index parameters.

If `compute_and_evaluate_ranking` wants to be used, you will also need:
- `ground_truth_path`: path to the ground truth of the benchmark.
- `k`: maximum value for the size of the rankings (e.g. Freyja's and Santos Small's benchmarks have k=10).
- `step`: how many joins to "jump" between evaluation points. For instance, given k = 20, we might want to evaluate the scores for every two increments of k (k = 2, k = 4, ..., k = 20). In this case `step` would be 2.

Once the set up is done, the code can be executed via the following command:
```
python 5_search.py
```