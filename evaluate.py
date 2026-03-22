import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from model import ResNetBiLSTM
from artist_data import ArtistDataset
from torchvision import transforms

# ===== Paths =====

task = "genre"   # change to genre if needed

image_root = "wikiart"
val_csv = f"wikiart_csv/{task}_val_clean.csv"
class_file = f"wikiart_csv/{task}_class.txt"
model_path = f"checkpoints/{task}_best_model.pt"

# ===== Device =====

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Classes =====

class_names = [line.strip() for line in open(class_file)]
num_classes = len(class_names)

# ===== Transform =====

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ===== Dataset =====

val_dataset = ArtistDataset(val_csv, image_root, transform)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

# ===== Model =====

model = ResNetBiLSTM(num_artists=num_classes).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))

model.eval()

# ===== Evaluation =====

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in tqdm(val_loader,desc="Evaluating"):

        images = images.to(device)

        outputs = model(images)["artist"]

        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# ===== Metrics =====

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=class_names
    )
)

print("\nConfusion Matrix:\n")

cm = confusion_matrix(all_labels, all_preds)

print(cm)