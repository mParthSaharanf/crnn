import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from artist_data import ArtistDataset
from model import ResNetBiLSTM
import torchvision.transforms as transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

dataset = ArtistDataset(
    "wikiart_csv/artist_val_clean.csv",
    "wikiart",
    transform
)

loader = DataLoader(dataset, batch_size=32)

model = ResNetBiLSTM(num_artists=23).to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pt"))
model.eval()

embeddings = []
labels = []

with torch.no_grad():
    for images, lbl in tqdm(loader):
        images = images.to(device)

        outputs = model(images)
        emb = outputs["embedding"]

        embeddings.append(emb.cpu().numpy())
        labels.append(lbl.numpy())

embeddings = np.concatenate(embeddings)
labels = np.concatenate(labels)

np.save("embeddings.npy", embeddings)
np.save("labels.npy", labels)

print("Embeddings saved")