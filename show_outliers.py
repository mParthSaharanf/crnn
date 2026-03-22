import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics.pairwise import cosine_distances

embeddings = np.load("embeddings.npy")
labels = np.load("labels.npy")

csv = pd.read_csv("wikiart_csv/artist_val.csv", header=None)
csv.columns = ["image_path","label"]

image_root = "wikiart"

outliers = []

for c in np.unique(labels):

    idx = np.where(labels == c)[0]
    class_emb = embeddings[idx]

    centroid = class_emb.mean(axis=0)

    distances = cosine_distances(class_emb, centroid.reshape(1,-1)).flatten()

    threshold = np.percentile(distances,95)

    for i,d in zip(idx,distances):
        if d > threshold:
            outliers.append((i,d))

# sort by most extreme
outliers = sorted(outliers,key=lambda x:-x[1])[:12]

plt.figure(figsize=(12,8))

for i,(idx,dist) in enumerate(outliers):

    row = csv.iloc[idx]
    img_path = f"{image_root}/{row.image_path}"

    img = Image.open(img_path)

    plt.subplot(3,4,i+1)
    plt.imshow(img)
    plt.title(f"dist={dist:.2f}")
    plt.axis("off")

plt.suptitle("Top Outlier Paintings")
plt.tight_layout()
plt.savefig("outliers.png",dpi=300)
plt.show()