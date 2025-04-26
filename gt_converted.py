
import os
import sys

import random
from multiprocessing import Pool

from tqdm import tqdm

from hgraph import HierVAE, PairVocab, MolGraph, common_atom_vocab

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rdkit import Chem

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

def encode_smiles_to_latent(model, smiles_list, vocab):
    """
    Encode valid SMILES strings into latent vectors.
    Invalid SMILES or encoding failures are skipped.
    
    Returns:
        latent_vectors: Tensor of latent vectors.
        valid_smiles: List of successfully encoded SMILES.
        valid_indices: Original indices of valid SMILES in the input list.
    """
    model.eval()
    latent_vectors = []
    valid_smiles = []
    valid_indices = []

    def safe_to(tensors, device):
        return [t.to(device) if hasattr(t, "to") else t for t in tensors]

    device = model.R_mean.weight.device

    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        print(idx)
        if mol is None:
            print(f"[skip] Invalid SMILES: {smiles}")
            continue

        try:
            _, tensors, _ = MolGraph.tensorize([smiles], vocab, common_atom_vocab)
            tree_tensors, graph_tensors = tensors
            tree_tensors = safe_to(tree_tensors, device)
            graph_tensors = safe_to(graph_tensors, device)

            with torch.no_grad():
                hroot, *_ = model.encoder(tree_tensors, graph_tensors)
                z, _ = model.rsample(hroot, model.R_mean, model.R_var, perturb=False)
                latent_vectors.append(z.squeeze(0))
                valid_smiles.append(smiles)
                valid_indices.append(idx)

        except Exception as e:
            print(f"[fail] Encoding failed for {smiles}: {e}")

    if not latent_vectors:
        raise RuntimeError("No valid SMILES were encoded.")

    return torch.stack(latent_vectors), valid_smiles, valid_indices

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

def train_mlp(latent_vectors, targets, input_size, hidden_size, epochs=100, lr=1e-3,
              weight_decay=1e-5, early_stop_patience=10):
    
    # Standardize X and y
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    latent_np = latent_vectors.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy().reshape(-1, 1)

    latent_std = scaler_X.fit_transform(latent_np)
    target_std = scaler_y.fit_transform(target_np).squeeze()

    X_train, X_val, y_train, y_val = train_test_split(
        latent_std, target_std, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    model = MLP(input_size, hidden_size, 1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train).squeeze()
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val).squeeze()
            val_loss = loss_fn(val_preds, y_val)

        # Save training history
        history.append({
            "epoch": epoch + 1,
            "train_loss": loss.item(),
            "val_loss": val_loss.item()
        })

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

        # Early stopping logic
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, scaler_X, scaler_y, history

def preprocess_and_encode_csv(model, csv_path, vocab, target_column="preds", smiles_column="smiles",
                               n_samples=60000, n_bins=20, random_state=42):
    """
    Full preprocessing pipeline:
    - Loads a large CSV with SMILES and target values.
    - Reduces to n_samples using stratified sampling on the target.
    - Encodes SMILES into latent vectors, filtering out invalid ones.
    
    Returns:
        latent_vectors: Tensor of encoded latent vectors.
        filtered_targets: Tensor of matching target values.
        valid_smiles: List of successfully encoded SMILES.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Step 1: Load and reduce
    df = pd.read_csv(csv_path)
    df["bin"] = pd.qcut(df[target_column], q=n_bins, duplicates="drop")
    df_sampled, _ = train_test_split(df, train_size=n_samples, stratify=df["bin"], random_state=random_state)
    df_sampled = df_sampled.drop(columns="bin").reset_index(drop=True)

    # Step 2: Encode SMILES
    smiles_list = df_sampled[smiles_column].tolist()
    target_list = df_sampled[target_column].tolist()

    latents, valid_smiles, valid_indices = encode_smiles_to_latent(model, smiles_list, vocab)
    filtered_targets = torch.tensor([target_list[i] for i in valid_indices], dtype=torch.float32)

    return latents, filtered_targets, valid_smiles

def decode_latent_to_smiles(model, latent_vector, max_decode_step=80):
    """
    Safely decode a latent vector into a SMILES string.
    Skips invalid or oversized generations.

    Returns:
        smiles (str) or None if decoding failed or SMILES invalid.
    """
    try:
        model.eval()
        with torch.no_grad():
            latent_vector = latent_vector.to(next(model.parameters()).device)
            smiles_list = model.decoder.decode(
                (latent_vector.unsqueeze(0),) * 3,
                greedy=True,
                max_decode_step=max_decode_step
            )
            smiles = smiles_list[0]
            mol = Chem.MolFromSmiles(smiles)
            return smiles if mol else None
    except Exception as e:
        print(f"[skip] Decoding failed: {e}")
        return None

def batch_optimize_latents(model, mlp, initial_latents, steps=1000, lr=1e-2, max_decode_step=50):
    """
    Batch optimize latent vectors and decode resulting molecules.

    Returns:
        optimized_smiles: List of decoded SMILES or None.
    """
    optimized_smiles = []

    for i, latent in enumerate(initial_latents):
        print(f"\n--- Optimizing vector {i+1}/{len(initial_latents)} ---")
        optimized_latent = optimize_latent_space(model, mlp, latent, steps=steps, lr=lr)

        # Try decoding
        smiles = decode_latent_to_smiles(model, optimized_latent, max_decode_step=max_decode_step)
        if smiles is not None:
            print(f"[ok] Decoded SMILES: {smiles}")
        else:
            print("[skip] Invalid or failed decoding.")

        optimized_smiles.append(smiles)

    return optimized_smiles

def optimize_latent_space(model, mlp, latent_vector, steps=1000, lr=1e-2, clamp_range=(-4, 4)):
    """
    Optimize a latent vector to maximize the target value.
    Clamps values to stay in valid range.

    Returns:
        optimized_latent: Final optimized latent vector (detached).
    """
    latent_vector = latent_vector.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([latent_vector], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        prediction = mlp(latent_vector)
        loss = -prediction  # We maximize the prediction
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            latent_vector.data.clamp_(*clamp_range)

        if not torch.isfinite(latent_vector).all():
            print(f"[stop] Non-finite values in latent vector at step {step}")
            return latent_vector.detach()

        print(f"Step {step + 1}/{steps}, Target Value: {-loss.item():.4f}")

    return latent_vector.detach()


seed=42
torch.manual_seed(seed)
random.seed(seed)

# Load vocabulary
path_to_vocab = "hgraph2graph/data/chembl/vocab.txt"
vocabu = [x.strip("\r\n ").split() for x in open(path_to_vocab)]
vocabu = PairVocab(vocabu)

# Initialize the model
class Args:
    vocab = vocabu
    atom_vocab = common_atom_vocab
    rnn_type = 'LSTM'
    hidden_size = 250
    embed_size = 250
    latent_size = 32
    depthT = 15
    depthG = 15
    diterT = 1
    diterG = 3
    dropout = 0.0

args = Args()
# Load pre-trained model

args = Args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HierVAE(args).to(device)

# Load the pre-trained model checkpoint
path_to_model_ckpt = "hgraph2graph/ckpt/chembl-pretrained/model.ckpt"
checkpoint = torch.load(path_to_model_ckpt, map_location=device)
model.load_state_dict(checkpoint[0])
model.eval()

path_to_data_to_train_MLP = "data/pseudo_labeled.csv"


latents, targets, smiles = preprocess_and_encode_csv(
    model,
    csv_path=path_to_data_to_train_MLP,
    vocab=vocabu,
    target_column="preds",
    smiles_column="SMILES",
    n_samples=10000
)



np.savez("latents_targets.npz", latents=latents.cpu().numpy(), targets=targets.cpu().numpy(), smiles=smiles)

data = np.load("latents_targets.npz", allow_pickle=True)
latents = torch.tensor(data["latents"], dtype=torch.float32)
targets = torch.tensor(data["targets"], dtype=torch.float32)
smiles = data["smiles"].tolist()

input_size = latents.size(1)
hidden_size = 128
mlp, scaler_X, scaler_y, history = train_mlp(
    latent_vectors=latents,
    targets=targets,
    input_size=latents.shape[1],
    hidden_size=128,
    epochs=1100,
    lr=1e-3,
    weight_decay=1e-5,
    early_stop_patience=10
)

random_indices = torch.randperm(len(targets))[:300]
top = latents[random_indices]

optimized_smiles = batch_optimize_latents(
    model=model,
    mlp=mlp,
    initial_latents=top,
    steps=1500,
    lr=0.01
)

optimized_smiles = [smiles for smiles in optimized_smiles if smiles is not None]

df = pd.DataFrame(optimized_smiles, columns=["SMILES"])

df.to_csv("optimized_smiles.csv", index=False)

