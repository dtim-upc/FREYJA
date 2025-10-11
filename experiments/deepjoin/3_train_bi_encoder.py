import os
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

def load_pairs(path: str, threshold: float, max_samples: int = None) -> pd.DataFrame:
    """Load and filter positive text pairs from CSV."""
    df = pd.read_csv(path)
    df = df[df["containment"] > threshold]
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(max_samples, random_state=0)
    return df[["col_text1", "col_text2"]]

def create_dataloader(df: pd.DataFrame, batch_size: int) -> DataLoader:
    """Convert dataframe to SentenceTransformer InputExamples and DataLoader."""
    examples = [InputExample(texts=[r.col_text1, r.col_text2]) for r in df.itertuples(index=False)]
    return DataLoader(examples, shuffle=True, batch_size=batch_size, drop_last=True)

def determine_training_params(dataset_size: int, batch_size: int):
    """Determine number of epochs and warmup steps based on dataset size."""
    if dataset_size < 5000:
        epochs = 15
    elif dataset_size < 50000:
        epochs = 10
    else:
        epochs = 5

    steps_per_epoch = dataset_size // batch_size
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(100, int(0.1 * total_steps))  # at least 100 steps

    return epochs, warmup_steps

def train_model(train_dataloader: DataLoader, model_name: str, output_dir: str, epochs: int, warmup_steps: int):
    """Train a SentenceTransformer model using MultipleNegativesRankingLoss."""
    model = SentenceTransformer(model_name)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': 2e-5, 'weight_decay': 0.01},
        output_path=output_dir,
        show_progress_bar=True,
    )
    print(f"Model saved to {output_dir}")
    return model

benchmark_name = "omnimatch_culture_recreation"
train_path = f"data/{benchmark_name}/train_data.csv"
output_dir = f"models/{benchmark_name}"
positive_threshold = 0.7
base_model = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(output_dir, exist_ok=True)
get_device()

df = load_pairs(train_path, threshold=positive_threshold)
print(f"Loaded {len(df)} positive pairs (threshold={positive_threshold}).")

batch_size = 32
train_dataloader = create_dataloader(df, batch_size)
epochs, warmup_steps = determine_training_params(len(df), batch_size)

train_model(train_dataloader, base_model, output_dir, epochs, warmup_steps)
