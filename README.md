# crnn
This repository implements a Convolutional Recurrent Neural Network (CRNN) to classify artwork and extract rich visual embeddings. The architecture leverages a pre-trained **ResNet-50 backbone**, a **BiLSTM** for sequential feature processing, and **Attention Pooling** to generate robust representations of paintings.
Because the model generates high-quality intermediate embeddings, it is highly effective for transfer learning, t-SNE clustering, and anomaly/outlier detection using cosine similarity.

## Repository Structure
* `model.py` - Core PyTorch `ResNetBiLSTM` architecture.
* `artist_data.py` - Custom PyTorch Dataset for loading WikiArt images and CSV labels.
* `artist_train.py` - Base training script for artist classification.
* `transfer_train.py` - Transfer learning script to adapt the base model to new tasks (e.g., genre or style classification).
* `extract_embeddings.py` - Generates and saves `.npy` feature embeddings from the trained model.
* `visualize_embeddings.py` - Performs t-SNE dimensionality reduction for cluster visualization.
* `show_outliers.py` - Calculates cosine distance from class centroids to flag and plot anomalous paintings.
* `evaluate.py` - Runs validation, calculates accuracy, and generates a confusion matrix.

## Usage
### 1. Base Training (Artist Classification)
Train the model from scratch to classify artists. This will create a `checkpoints/best_model.pt` file.
### 2. Extracting Embeddings & Finding Similarities
