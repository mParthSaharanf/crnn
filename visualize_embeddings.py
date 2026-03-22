import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

embeddings = np.load("embeddings.npy")
labels = np.load("labels.npy")

print("Running t-SNE (this may take a minute)...")

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
emb_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(10,8))

scatter = plt.scatter(
    emb_2d[:,0],
    emb_2d[:,1],
    c=labels,
    cmap="tab20",
    alpha=0.6
)

plt.colorbar(scatter)
plt.title("Artist Embedding Clusters (t-SNE)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

plt.savefig("artist_clusters_tsne.png", dpi=300)
plt.show()