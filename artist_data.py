import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class ArtistDataset(Dataset):
    def __init__(self, csv_file, image_root, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_root, row['image_path'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row['label'], dtype=torch.long)
        return image, label