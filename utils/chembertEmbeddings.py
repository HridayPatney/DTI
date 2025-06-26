import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Load ChemBERTa for SMILES
smiles_model_name = "seyonec/ChemBERTa-zinc-base-v1"
tokenizer = AutoTokenizer.from_pretrained(smiles_model_name)
model = AutoModel.from_pretrained(smiles_model_name)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Embedding function for SMILES
def get_smiles_embedding(smiles):
    if pd.isna(smiles): return []
    smiles = str(smiles).strip()
    with torch.no_grad():
        inputs = tokenizer(smiles, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()
    return embedding

# Process CSV (keeping protein_embedding as-is)
def process_csv(input_path, output_path):
    df = pd.read_csv(input_path)

    smiles_embeddings = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        smiles_embeddings.append(get_smiles_embedding(row["smiles"]))

    df["smiles_embedding"] = [str(x) for x in smiles_embeddings]
    df.to_csv(output_path, index=False)

# Run it
input_csv = "Davis.csv"  # contains precomputed protein_embedding
output_csv = "DavisChembert.csv"
process_csv(input_csv, output_csv)