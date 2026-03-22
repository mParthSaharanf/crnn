class MultiTaskDataset(Dataset):

    def __init__(self, dataframe, image_root, transform):

        self.df = dataframe
        self.image_root = image_root
        self.transform = transform

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_path = os.path.join(self.image_root, row["image"])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return (
            image,
            row["artist"],
            row["style"],
            row["genre"]
        )