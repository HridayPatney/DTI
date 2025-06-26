import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm

# Load ProtBert tokenizer and model
model_name = "Rostlab/prot_bert"
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
model = AutoModel.from_pretrained(model_name)

# Send model to device once
device = "cpu"  # or "cuda" or "cpu"
model.to(device)
model.eval()

def get_protein_embedding(sequence, chunk_size=510):
    sequence = sequence.strip().upper()
    chunks = [sequence[i:i+chunk_size] for i in range(0, len(sequence), chunk_size)]
    chunk_embeddings = []

    with torch.no_grad():
        for chunk in chunks:
            chunk_spaced = " ".join(list(chunk))
            encoded_input = tokenizer(chunk_spaced, return_tensors="pt", truncation=True, max_length=chunk_size+2)
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            output = model(**encoded_input)
            chunk_emb = output.last_hidden_state.mean(dim=1)  # Mean pooling
            chunk_embeddings.append(chunk_emb)

    final_embedding = torch.mean(torch.stack(chunk_embeddings), dim=0)
    return final_embedding.cpu().numpy().flatten()

def process_csv(input_csv, output_csv, chunk_size=510):
    df = pd.read_csv(input_csv)
    embeddings = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sequence = row["sequence"]
        embedding = get_protein_embedding(sequence, chunk_size=chunk_size)
        embedding_str = ",".join(map(str, embedding))
        embeddings.append(embedding_str)

    df["protein_embedding"] = embeddings
    df.to_csv(output_csv, index=False)
    print(f"Saved output CSV with embeddings to: {output_csv}")

# File paths
input_csv = "DavisChembert.csv"
output_csv = "Davis_protbert_chembert.csv"

# Run the processing
process_csv(input_csv, output_csv, chunk_size=510)
