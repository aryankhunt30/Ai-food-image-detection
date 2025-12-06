import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from model import FoodClassifier
from dataset import create_dataloaders
import warnings
warnings.filterwarnings("ignore", message="Error fetching version info")


def train(
    h5_path,
    batch_size=32,
    epochs=10,
    lr=1e-4,
):

    train_transform = None  # you can add Albumentations here
    val_transform = None

    train_loader, val_loader = create_dataloaders(
        h5_path=h5_path,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=batch_size,
    )

    model = FoodClassifier(num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        pbar = tqdm(train_loader)
        total_loss = 0

        for images, labels in pbar:
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_description(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} Avg loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "model_food.pth")
    print("Model saved: model_food.pth")


# -----------------------------------------------------------
# 🚀 IMPORTANT: CALL TRAINING HERE
# -----------------------------------------------------------
if __name__ == "__main__":
    train(
        h5_path="food_c101_n1000_r384x384x3.h5",
        batch_size=32,
        epochs=5,
        lr=1e-4
    )