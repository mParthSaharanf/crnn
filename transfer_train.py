import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import ResNetBiLSTM
from artist_data import ArtistDataset


def load_pretrained_exclude_classifier(model, checkpoint_path, device):

    print(f"\nLoading pretrained weights from {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model_dict = model.state_dict()

    filtered_dict = {
        k: v for k, v in state_dict.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

    print(f"Loaded {len(filtered_dict)} layers")

    return model


def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()
    running_loss, correct, total = 0, 0, 0

    for images, labels in tqdm(loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)["artist"]

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(1)

        correct += (preds == labels).sum().item()

        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):

    model.eval()

    running_loss, correct, total = 0, 0, 0

    with torch.no_grad():

        for images, labels in tqdm(loader):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)["artist"]

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = outputs.argmax(1)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

    return running_loss / total, correct / total


def main():

    # ===== Choose task =====
    task = "genre"   # change to "genre"

    image_root = "wikiart"

    train_csv = f"wikiart_csv/{task}_train_clean.csv"
    val_csv = f"wikiart_csv/{task}_val_clean.csv"
    class_file = f"wikiart_csv/{task}_class.txt"

    pretrained_artist_model = "checkpoints/best_model.pt"

    num_classes = len(open(class_file).readlines())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    train_dataset = ArtistDataset(train_csv, image_root, transform)
    val_dataset = ArtistDataset(val_csv, image_root, transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # ===== Model =====

    model = ResNetBiLSTM(num_artists=num_classes).to(device)

    # ===== Load artist weights =====

    model = load_pretrained_exclude_classifier(
        model,
        pretrained_artist_model,
        device
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    num_epochs = 20
    best_loss = float("inf")

    patience = 3
    epochs_no_improve = 0

    for epoch in range(num_epochs):

        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        print(
            f"Train Loss {train_loss:.4f}  Train Acc {train_acc:.4f}"
        )

        print(
            f"Val Loss {val_loss:.4f}  Val Acc {val_acc:.4f}"
        )

        if val_loss < best_loss:

            best_loss = val_loss
            epochs_no_improve = 0

            torch.save(
                model.state_dict(),
                f"checkpoints/{task}_best_model.pt"
            )

            print("Saved best model")
        else:
            epochs_no_improve +=1
            print(f"No improvement ({epochs_no_improve}/{patience})")
        
        if epochs_no_improve >= patience:
            print("early stopping triggered")
            break


if __name__ == "__main__":
    main()