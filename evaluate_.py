import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from model import ResNetBiLSTM
from artist_data import ArtistDataset
from torchvision import transforms

# ===== Paths =====

task = "genre"

image_root = "wikiart"
val_csv = f"wikiart_csv/{task}_val_clean.csv"
class_file = f"wikiart_csv/{task}_class.txt"
model_path = f"checkpoints/{task}_best_model.pt"

# ===== Device =====

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===== Classes =====

class_names = [line.strip() for line in open(class_file)]
num_classes = len(class_names)

print("Number of classes:", num_classes)

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

print("Validation samples:", len(val_dataset))

# ===== Model =====

model = ResNetBiLSTM(num_artists=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

model.eval()

# ===== Evaluation =====

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in tqdm(val_loader, desc="Evaluating"):

        images = images.to(device)

        outputs = model(images)["artist"]

        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# ===== Metrics =====

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================\n")

print(classification_report(
    all_labels,
    all_preds,
    target_names=class_names
))

acc = accuracy_score(all_labels, all_preds)

print("Overall Accuracy:", acc)

# ===== Confusion Matrix =====

cm = confusion_matrix(all_labels, all_preds)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================\n")

print(cm)

# ===== Plot Confusion Matrix =====

plt.figure(figsize=(12,10))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig("confusion_matrix.png")
plt.show()