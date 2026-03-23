import numpy as np
from sklearn.metrics.pairwise import cosine_distances

embeddings = np.load("embeddings.npy")
labels = np.load("labels.npy")

num_classes = len(np.unique(labels))

outliers = []

for c in range(num_classes):

    idx = np.where(labels == c)[0]
    class_emb = embeddings[idx]

    centroid = class_emb.mean(axis=0)

    distances = cosine_distances(class_emb, centroid.reshape(1,-1)).flatten()

    threshold = np.percentile(distances, 95)

    for i, d in zip(idx, distances):
        if d > threshold:
            outliers.append((i, c, d))

print("Top Outliers:")
for o in sorted(outliers, key=lambda x: -x[2])[:20]:
    print(o)