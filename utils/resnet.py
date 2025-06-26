import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm

# ========== Config ==========
csv_path = "Davis_protbert_chembert.csv"         # Replace with your dataset file
image_dir = "mol_images2"              # Directory to save molecule images
output_csv = "final_dataset_with_Davis_new.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 224
batch_size = 64

# ========== Step 1: Load dataset ==========
df = pd.read_csv(csv_path)
os.makedirs(image_dir, exist_ok=True)

# ========== Step 2: Generate RDKit images ==========
image_paths = []
for idx, smiles in tqdm(enumerate(df['smiles']), total=len(df), desc="Generating molecule images"):
    mol = Chem.MolFromSmiles(smiles)
    img_name = f"mol_{str(idx).zfill(6)}.png"
    img_path = os.path.join(image_dir, img_name)

    if mol is not None:
        try:
            Draw.MolToFile(mol, img_path)
            image_paths.append(img_path)
        except Exception as e:
            print(f"Draw error at index {idx}: {e}")
            image_paths.append(None)
    else:
        print(f"Invalid SMILES at index {idx}: {smiles}")
        image_paths.append(None)

df['image_path'] = image_paths

# ========== Step 3: Dataset for loading images ==========
class MoleculeImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        if img_path is None or not os.path.exists(img_path):
            return torch.zeros(3, image_size, image_size)  # Blank if image failed
        img = Image.open(img_path).convert('RGB')
        return self.transform(img)

dataset = MoleculeImageDataset(df['image_path'].tolist())
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# ========== Step 4: Load ResNet model ==========
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove classification head
resnet.to(device)
resnet.eval()

# ========== Step 5: Extract ResNet features ==========
resnet_features = []

with torch.no_grad():
    for batch in tqdm(loader, desc="Extracting ResNet features"):
        batch = batch.to(device)
        features = resnet(batch)  # Shape: [B, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # Flatten
        resnet_features.extend(features.cpu().numpy())

# ========== Step 6: Add to DataFrame and save ==========
df['resnet_features'] = resnet_features
df['resnet_features'] = df['resnet_features'].apply(lambda x: ','.join(map(str, x)))
df.to_csv(output_csv, index=False)

print(f"✅ All done! Saved to: {output_csv}")
