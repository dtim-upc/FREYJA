import os
import pandas as pd
from tqdm import tqdm
import sys
import numpy as np
import torch  # needed for GPU support

sys.path.append(os.path.join(os.path.dirname(__file__), 'TaBERT'))
from TaBERT.table_bert import Table, Column
from TaBERT.table_bert.table_bert import TableBertModel

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pretrained TableBERT
model = TableBertModel.from_pretrained('model/tabert_large_k3/model.bin')
model.to(device)  # move model to GPU if available
model.eval()

datalake_path = "C:/Projects/santos_small/datalake"
embeddings_path = "embeddings/santos_small"
os.makedirs(embeddings_path, exist_ok=True)

all_embeddings = []
metadata = []

def column_embedding(col_name, column_values, table_id=None, context=None):
    if table_id is None:
        table_id = "table"
    if context is None:
        context = ""

    # Remove NaNs and empty strings
    non_null = column_values.dropna().astype(str)
    non_null = non_null[non_null.str.strip() != ""]

    if non_null.empty:
        return None  # Skip this column entirely

    sample_value = non_null.iloc[0]

    table = Table(
        id=table_id,
        header=[Column(col_name, 'text', sample_value=sample_value)],
        data=[[str(v)] for v in non_null]
    ).tokenize(model.tokenizer)

    context_encoding, column_encoding, info_dict = model.encode(
        contexts=[model.tokenizer.tokenize(context)],
        tables=[table]
    )

    tensor = column_encoding[0].to(device)
    if tensor.dim() == 3:
        cls_embedding = tensor[:, 0, :].squeeze(0).detach().cpu().numpy()
    else:
        cls_embedding = tensor[0, :].detach().cpu().numpy()

    return cls_embedding

# Iterate over the CSV files
count = 1
for filename in tqdm(os.listdir(datalake_path)):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(datalake_path, filename), encoding="latin1")

        for col in df.columns:
            emb = column_embedding(col, df[col], table_id=filename)
            if emb is not None:
                all_embeddings.append(emb)
                metadata.append({"table": filename, "column": col})
    if count % 100 == 0: # Save data every 100 files, in case of issues during the execution
        all_embeddings_np = np.stack(all_embeddings)  # Shape [N, d]

        metadata_df = pd.DataFrame(metadata)
        metadata_df.insert(0, "id", range(len(metadata_df)))  # Add index column

        np.save(os.path.join(embeddings_path, f"embeddings_iteration{count}.npy"), all_embeddings_np)
        metadata_df.to_csv(os.path.join(embeddings_path, f"column_metadata_iteration{count}.csv"), index=False)
    count = count + 1


# Save embeddings and metadata
all_embeddings = np.stack(all_embeddings)  # Shape [N, d]

metadata_df = pd.DataFrame(metadata)
metadata_df.insert(0, "id", range(len(metadata_df)))  # add index column

np.save(os.path.join(embeddings_path, f"embeddings.npy"), all_embeddings)
metadata_df.to_csv(os.path.join(embeddings_path, f"column_metadata.csv"), index=False)