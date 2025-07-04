# -*- coding: utf-8 -*-
"""notebook871629bead

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/hridaypatney05/notebook871629bead.3dfd3cbd-25e1-46de-867d-b4e52ed3a02b.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20250626/auto/storage/goog4_request%26X-Goog-Date%3D20250626T092926Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D95a3c6f73edc951ece457ce7fa7274eb691b5fd2070c110f8bcef66df26ebd40e0218e6bea272e1fecdeb5590a2debcde490994c907c8719ad4fc98350817ed8ebe271a85f9ba73bd68d360cd5aaa7eef4479f32ffc37b434dced44e4b37399d5164707fdd28e9e1df67f04aa1661bdbf973829df5527bac8dea66576b5f380a18ffebad8ba21cb3eb84a1505021da2a5728bc1416ffbba044090f014da614d3a1ebb9a9a6fc13a942cb5ca56675e0316b3a9fcb54f9fda4eddc1b04ff542dad07d702a06d1ec2bf70361fecd726e330c5db0fe45a22fef0c6d2e89d079844245be70300fd699e08459bb2c33dd01a8fdec7d4d0037c91253a274c4112d66e8f
"""

# IMPORTANT: SOME KAGGLE DATA SOURCES ARE PRIVATE
# RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES.
import kagglehub
kagglehub.login()

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

hridaypatney05_kibadataset_path = kagglehub.dataset_download('hridaypatney05/kibadataset')

print('Data source import complete.')

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import ast
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

class Config:
    def __init__(self):
        self.hidden_dim = 512
        self.dropout_rate = 0.2

class DTIDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        print(f"Dataset loaded with {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Convert string representations to tensors
        protein_embedding = torch.tensor(
            ast.literal_eval(row['protein_embedding']),
            dtype=torch.float32
        )  # [1024] from ProtBERT

        smiles_embedding = torch.tensor(
            ast.literal_eval(row['smiles_embedding']),
            dtype=torch.float32
        )  # [768] from ChemBERT

        resnet_features = torch.tensor(
            ast.literal_eval(row['resnet_features']),
            dtype=torch.float32
        )  # [2048] from ResNet50

        label = torch.tensor(row['label'], dtype=torch.long)

        return {
            'protein_embedding': protein_embedding,
            'smiles_embedding': smiles_embedding,
            'resnet_features': resnet_features,
            'label': label
        }

class SimplifiedMultimodalDTI(nn.Module):
    def __init__(self, config):
        super(SimplifiedMultimodalDTI, self).__init__()

        # Input dimensions from precomputed embeddings
        self.protein_dim = 1024  # ProtBERT
        self.smiles_dim = 768    # ChemBERT
        self.image_dim = 2048    # ResNet50

        # Feature transformation layers
        self.protein_transform = nn.Linear(self.protein_dim, config.hidden_dim)
        self.smiles_transform = nn.Linear(self.smiles_dim, config.hidden_dim)
        self.image_transform = nn.Linear(self.image_dim, config.hidden_dim)

        # Multi-head self-attention for image features
        self.msa_layer = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8
        )

        # Fusion layer for combining SMILES and image features
        self.fusion_layer = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

        # Attention mechanisms
        self.attention_layer = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.protein_attention_layer = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.drug_attention_layer = nn.Linear(config.hidden_dim, config.hidden_dim)

        # Dropout layers
        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.dropout2 = nn.Dropout(config.dropout_rate)
        self.dropout3 = nn.Dropout(config.dropout_rate)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()

        # Fully connected layers for final prediction
        self.fc1 = nn.Linear(config.hidden_dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

    def forward(self, protein_embedding, smiles_embedding, resnet_features):
        batch_size = protein_embedding.size(0)

        # Transform embeddings to common hidden dimension
        protein_features = self.relu(self.protein_transform(protein_embedding))
        smiles_features = self.relu(self.smiles_transform(smiles_embedding))
        image_features = self.relu(self.image_transform(resnet_features))

        # Process image features with attention
        image_features = image_features.unsqueeze(0)  # [1, batch, hidden_dim]
        attn_output, _ = self.msa_layer(image_features, image_features, image_features)
        image_features_attn = attn_output.squeeze(0)  # [batch, hidden_dim]

        # Combine drug SMILES and image features
        combined_drug_features = torch.cat([smiles_features, image_features_attn], dim=1)
        fused_drug_features = self.relu(self.fusion_layer(combined_drug_features))

        # Apply attention mechanism
        drug_att = self.drug_attention_layer(fused_drug_features)
        protein_att = self.protein_attention_layer(protein_features)

        # Create attention matrices
        d_att_expanded = drug_att.unsqueeze(2)  # [batch, hidden_dim, 1]
        p_att_expanded = protein_att.unsqueeze(1)  # [batch, 1, hidden_dim]

        # Calculate attention matrix
        attention_matrix = self.attention_layer(self.relu(d_att_expanded + p_att_expanded))

        # Calculate attention weights
        compound_attention = torch.mean(attention_matrix, dim=2)  # [batch, hidden_dim]
        protein_attention = torch.mean(attention_matrix, dim=1)   # [batch, hidden_dim]

        compound_attention = self.sigmoid(compound_attention)
        protein_attention = self.sigmoid(protein_attention)

        # Apply attention weights
        fused_drug_features = fused_drug_features * 0.5 + fused_drug_features * compound_attention
        protein_features = protein_features * 0.5 + protein_features * protein_attention

        # Concatenate drug and protein features
        pair = torch.cat([fused_drug_features, protein_features], dim=1)

        # Fully connected layers for prediction
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)

        return predict

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate all metrics from the comparison table"""
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    if torch.is_tensor(y_pred_proba):
        y_pred_proba = y_pred_proba.cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')

    if y_pred_proba.shape[1] == 2:
        y_pred_proba_pos = y_pred_proba[:, 1]
    else:
        y_pred_proba_pos = y_pred_proba

    auc = roc_auc_score(y_true, y_pred_proba_pos)
    aupr = average_precision_score(y_true, y_pred_proba_pos)

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'AUC': auc,
        'AUPR': aupr
    }

def evaluate_model(model, test_loader, device):
    """Evaluate model and calculate all metrics"""
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            protein_emb = batch['protein_embedding'].to(device)
            smiles_emb = batch['smiles_embedding'].to(device)
            resnet_feat = batch['resnet_features'].to(device)
            labels = batch['label'].to(device)

            outputs = model(protein_emb, smiles_emb, resnet_feat)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    metrics = calculate_metrics(all_labels, all_predictions, all_probabilities)
    return metrics, all_labels, all_predictions, all_probabilities

def split_and_create_data_loaders(csv_file, test_size=0.2, batch_size=32, num_workers=4, random_state=42):
    """Load CSV file and split into train/test sets, then create data loaders"""
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_type}")

    # Load the full dataset
    print(f"Loading dataset from {csv_file}...")
    full_data = pd.read_csv(csv_file)
    print(f"Full dataset loaded with {len(full_data)} samples")

    # Check class distribution
    print(f"Class distribution:")
    print(full_data['label'].value_counts())

    # Split the data into train and test sets
    train_data, test_data = train_test_split(
        full_data,
        test_size=test_size,
        random_state=random_state,
        stratify=full_data['label']  # Stratified split to maintain class balance
    )

    print(f"Train set: {len(train_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    print(f"Train/Test split ratio: {len(train_data)/len(full_data):.2f}/{len(test_data)/len(full_data):.2f}")

    # Reset indices for the split datasets
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    kwargs = {
        'num_workers': num_workers,
        'pin_memory': True if device_type == 'cuda' else False
    }

    # Create datasets
    train_dataset = DTIDataset(train_data)
    test_dataset = DTIDataset(test_data)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader

def train_and_evaluate(model, train_loader, test_loader, num_epochs=100, learning_rate=0.001):
    """Train the model and evaluate periodically with progress bars"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_metrics = None
    train_losses = []

    print("Starting training...")

    # Outer progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0.0

        # Inner progress bar for batches within each epoch
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch_idx, batch in enumerate(batch_pbar):
            protein_emb = batch['protein_embedding'].to(device)
            smiles_emb = batch['smiles_embedding'].to(device)
            resnet_feat = batch['resnet_features'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(protein_emb, smiles_emb, resnet_feat)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Update batch progress bar with current loss
            batch_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{train_loss/(batch_idx+1):.4f}'
            })

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'Avg Loss': f'{avg_train_loss:.4f}'
        })

        # Evaluation phase every 10 epochs
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f'\nEvaluating at epoch {epoch}...')
            metrics, _, _, _ = evaluate_model(model, test_loader, device)

            print(f'Epoch [{epoch}/{num_epochs}]')
            print(f'Train Loss: {avg_train_loss:.4f}')
            print(f'Test Metrics:')
            for metric_name, value in metrics.items():
                print(f'  {metric_name}: {value:.4f}')
            print('-' * 50)

            best_metrics = metrics

    return best_metrics, train_losses

def print_final_results(metrics):
    """Print results in the format similar to the comparison table"""
    print("\n" + "="*60)
    print("FINAL EVALUATION RESULTS")
    print("="*60)
    print(f"{'Metric':<15} {'Value':<10}")
    print("-"*25)
    for metric_name, value in metrics.items():
        print(f"{metric_name:<15} {value:.3f}")
    print("="*60)

    # Compare with HyperAttentionDTI baseline from the table
    print("\nComparison with HyperAttentionDTI baseline:")
    baseline = {
        'Accuracy': 0.866,
        'Precision': 0.754,
        'Recall': 0.780,
        'AUC': 0.920,
        'AUPR': 0.839
    }

    print(f"{'Metric':<15} {'Our Model':<12} {'Baseline':<12} {'Improvement':<12}")
    print("-"*55)
    for metric_name, our_value in metrics.items():
        baseline_value = baseline[metric_name]
        improvement = our_value - baseline_value
        print(f"{metric_name:<15} {our_value:.3f}      {baseline_value:.3f}      {improvement:+.3f}")

def plot_results(train_losses, metrics, y_true, y_proba):
    """Plot training loss and evaluation curves"""
    plt.figure(figsize=(15, 5))

    # Plot training loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    plt.subplot(1, 3, 2)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["AUC"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)

    # Plot Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba[:, 1])
    plt.subplot(1, 3, 3)
    plt.plot(recall_curve, precision_curve, label=f'PR Curve (AUPR = {metrics["AUPR"]:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the complete pipeline"""
    # Initialize configuration
    config = Config()

    # Initialize model
    model = SimplifiedMultimodalDTI(config)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Load data and create train/test split
    csv_file_path = '/kaggle/input/kibadataset/final_dataset_with_KIBA.csv'  # Replace with your actual CSV file path
    train_loader, test_loader = split_and_create_data_loaders(
        csv_file_path,
        test_size=0.2,  # 80% train, 20% test
        batch_size=64,
        num_workers=4,
        random_state=42  # For reproducible splits
    )

    # Train and evaluate the model
    final_metrics, train_losses = train_and_evaluate(
        model,
        train_loader,
        test_loader,
        num_epochs=100,
        learning_rate=0.001
    )

    # Print final results
    print_final_results(final_metrics)

    # Detailed evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics, y_true, y_pred, y_proba = evaluate_model(model, test_loader, device)

    # Print confusion matrix and classification report
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Plot results
    plot_results(train_losses, metrics, y_true, y_proba)

    # Save model
    torch.save(model.state_dict(), 'simplified_multimodal_dti_model.pth')
    print("Model saved as 'simplified_multimodal_dti_model.pth'")

if __name__ == "__main__":
    main()