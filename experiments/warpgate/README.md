# WarpGate
This repository contains our own implementation for [*WarpGate: A Semantic Join Discovery System for Cloud Data Warehouses*](https://arxiv.org/abs/2212.14155). The paper proposes a system designed to help users of cloud data warehouses (CDWs) discover tables (or columns) that can be joined with a given table / column, both via exact syntactic matches and via semantic joinability (i.e. even if formats differ or names don’t exactly match). 

WarpGate works by embedding table columns into a high-dimensional vector space, so that semantically similar / joinable columns end up close together. It also uses locality-sensitive hashing (SimHash) to index those embeddings to allow fast approximate nearest neighbor lookup. Since data warehouses often contain huge tables, the paper addresses efficiency by sampling (so you don’t need to read entire tables) and shows that embeddings from sampled data still preserve enough signal to make good join suggestions. They integrate the system into a BI product (Sigma Workbooks), where users can request top-k candidate joinable columns through a “Add column via lookup” feature. The evaluation shows WarpGate is better than several baselines in precision/recall, is sample efficient, and scales to large tables.

WarpGate acts as a relevant baseline for our analyses, given that it was one of the first systems to employ an embedding-based approach for the task of join discovery. Unfortunately, no open implementation is available. Hence, we developed our own version following the indications of the original paper.

## Implementation Details
WarpGate's mechanism follows a straightforward, two-step process:
1. Embedding generation: Produce embeddings of the tables using models specifically designed for tabular data (**without fine-tuning**).
2. Indexing with LSH: Store these embeddings in a locality-sensitive hashing (LSH) index for efficient retrieval.

At query time, the index is traversed to select the closest elements to the query (i.e. those in the corresponding LSH bucket), and a similarity search is performed on this reduced subset of potential joins.

The original WarpGate paper does not specify the embedding model used but outlines three criteria to select it:
1. The model should be pre-trained on **tabular** data
2. The model should trained on a **large corpora**
3. The model should have a reasonable inference time (i.e. the embeddings should not be exceedingly costly to compute). 

Several models meet these requirements, including [Tabbie](https://arxiv.org/abs/2105.02584), [Turl](https://arxiv.org/abs/2006.14806), [TaPas](https://arxiv.org/abs/2004.02349) and [TaBERT](https://arxiv.org/abs/2005.08314). We selected TaBERT because, despite slightly slower inference, it serializes the table (rows and columns) together with associated natural language text (e.g., table descriptions, queries or surrounding web text). This enables the model to connect column headers and cell values with linguistic meaning (e.g., “capital” ↔ “city name,” “GDP” ↔ “economy”). Additionally, it was trained on millions of real web tables aligned with natural language, providing embeddings that better capture the underlying properties of columns. We employed [TaBERT-large-K3](https://github.com/facebookresearch/TaBERT) with the default parameters.

Once embeddings are generated, we build the LSH index, which buckets embeddings using a family of locality-sensitive hash functions, creating natural neighbor groups. At inference, the query column is hashed, closest neighbors are retrieved and similarity is computed only among them. We configured the LSH index with standard parameters: 16 hash hyperplanes, 8 tables and multiprobe activated. Similarity is assessed via cosine similarity, so embeddings are normalized before storage.

## Execution Instructions

### Requirements

#### TaBERT model
As stated above, we use the [TaBERT-large-K3](https://github.com/facebookresearch/TaBERT) version. The indicated link points to the TaBERT github page. There, you can find a section with *pre-trained models*, pointing to a Google Drive link. Access the link and download the *tabert_base_large_k3.tar.gz* file (if you want to employ other versions, there should be no issue, although the results might vary).

In this repository you will find a *models* folder. Unzip the downloaded file there. You should be left with a folder with the name *tabert_large_k3*.

You will also note a *TaBERT* folder in the repository. This folder contains a reduced version of the original TaBERT code, with only the required elements to develop our task.

#### Python libraries
The recommended Python version is 3.9.

The required set of libraries is contained in the `requirements.txt` file. You can install them with the following command:
```
pip install -r requirements.txt
```
To prevent dependency issues, it is recommended to create a virtual environment using conda.

Additionally, you will have to install `torch_scatter`, a PyTorch extension used for efficient tensor aggregation (scatter operations):
- If you want to use CPU only, run the following command (note that the `torch` version in `requirements.txt` is `2.3`):
```
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
```
- If you want to use GPU, you will need to check which CUDA version is supported by your GPU and then run the following command (you can check this by running `nvidia-smi`). For CUDA 1.21, it would be:
```
pip uninstall -y torch torch-scatter

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```
The code is prepared to use CUDA environments if these exist.

## Step 1: Creating Embeddings
The first step involved the generation of the embeddings for all the columns in the data lake. The necessary code is in file `1_create_embeddings.py`. There are two variables to modify:
- `datalake_path`: path to the folder where the data is contained. This should be a list of CSV files.
- `embeddings_apth`: directory to store the embeddings. By default, it is located in a folder in the same directory, divided in subfolder by benchmarks.

Additionally, make sure that the loading of the model is done properly (that is, that `TableBertModel.from_pretrained()` loads the `model.bin` of the downloaded TaBERT model). If available, GPU will be used to perform the generation.

Once the set up is done, the code can be executed via the following command:
```
python 1_create_embeddings.py
```
There might be some additional libraries that might need to be downloaded the first time the code is executed.

The code will iterate over all the CSV files in `datalake_path` and generate the corresponding embeddings per column. Note that we remove NaN values and empty strings to prevent problems with TaBERT. The output will be composed of two files:
- `column_metadata.csv`: CSV file that indicates an index for each column. This will be used to associate columns to embeddings in the retrieval process.
- `embeddings.npy`: resulting embeddings per column, stored in a NumPy binary file to store the arrays efficiently. The embeddings also contain the index indicated in `column_metadata.csv`.

Note that, by default, these files are generated every 100 CSV files processed. Hence, if there is some error during the generation of the embeddings, there is no need to repeat the entire process.


## Step 2: Semantic Search
Once the embeddings have been obtained, we can perform similarity search to detect similar columns; file `2_search.py`.

In this file you will find an `LSH` index class, used to build an LSH index as specified in the original paper. The index parameters are the recommended for such an index.

You will also find the `compute_and_evaluate_ranking` function. This function is employed to obtain precision, recall (raw, maximum possible and percentual) and MAP scores, as those observed in the paper. If you do not want to use this function, an example of a query is also attached (i.e. simply querying the index and checking the results).

In both cases, you will need to define the following parameter:
- `datalake_name`: assuming that you are using the default directory for the data generated in the previous script, this name is used to automatically obtain the embeddings and metadata needed.

If `compute_and_evaluate_ranking` wants to be used, you will also need:
- `ground_truth_path`: path to the ground truth of the benchmark.
- `k`: maximum value for the size of the rankings (e.g. Freyja's and Santos Small's benchmarks have k=10).
- `step`: how many joins to "jump" between evaluation points. For instance, given k = 20, we might want to evaluate the scores for every two increments of k (k = 2, k = 4, ..., k = 20). In this case `step` would be 2.

Once the set up is done, the code can be executed via the following command:
```
python 2_search.py
```