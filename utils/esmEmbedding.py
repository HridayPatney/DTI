import torch
import esm
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast

def compute_esm2_embeddings(csv_file, output_file, batch_size=8):
    """
    Pre-compute ESM-2 embeddings for all protein sequences and save to CSV
    """
    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load your dataset
    df = pd.read_csv(csv_file)
    print(f"Processing {len(df)} protein sequences...")
    
    # Store embeddings
    protein_embeddings = []
    
    # Process in batches
    for i in tqdm(range(0, len(df), batch_size), desc="Computing ESM-2 embeddings"):
        batch_end = min(i + batch_size, len(df))
        batch_sequences = df.iloc[i:batch_end]['sequence'].tolist()
        
        # Prepare batch data for ESM-2
        batch_data = [(f"protein_{j}", seq) for j, seq in enumerate(batch_sequences)]
        
        try:
            # Convert to ESM-2 format
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)
            
            # Get embeddings
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
                token_representations = results["representations"][33]
            
            # Extract sequence-level representations (mean pooling)
            for j, (_, seq) in enumerate(batch_data):
                # Skip special tokens (BOS and EOS)
                seq_len = len(seq)
                seq_repr = token_representations[j, 1:seq_len+1].mean(0)
                protein_embeddings.append(seq_repr.cpu().numpy().tolist())
                
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {e}")
            # Add empty embeddings for failed sequences
            for _ in range(batch_end - i):
                protein_embeddings.append([0.0] * 1280)  # ESM-2 650M has 1280 dimensions
    
    # Add embeddings to dataframe
    df['protein_embedding'] = protein_embeddings
    
    # Save updated dataframe
    df.to_csv(output_file, index=False)
    print(f"Embeddings saved to {output_file}")
    
    return df

# Usage
if __name__ == "__main__":
    input_csv = "final_dataset_with_resnet.csv"  # Your original dataset
    output_csv = "dataset_with_esm2_embeddings.csv"  # Output with embeddings
    
    df_with_embeddings = compute_esm2_embeddings(input_csv, output_csv, batch_size=4)
