import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import FoodH5Dataset
from model import FoodClassifier

# -----------------------------
# Device Selection
# -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# -----------------------------
# Load Model
# -----------------------------
model = FoodClassifier(num_classes=3)
model.load_state_dict(torch.load("model_food.pth", map_location="cpu"))
model = model.to(device)
model.eval()

# -----------------------------
# Load Dataset
# -----------------------------
full_dataset = FoodH5Dataset(
    "/Users/aryankhunt/Downloads/Food detection/food_c101_n1000_r384x384x3.h5",
    transform=None
)
subset = torch.utils.data.Subset(full_dataset, range(200))
loader = DataLoader(subset, batch_size=64, shuffle=False)

# -----------------------------
# Accuracy Per Batch
# -----------------------------
batch_accuracies = []
batch_numbers = []

with torch.no_grad():
    total_correct = 0
    total_seen = 0
    batch_id = 0

    for images, labels in loader:
        outputs = model(images.to(device))
        _, preds = torch.max(outputs.cpu(), 1)

        correct = (preds == labels).sum().item()
        total = labels.size(0)

        total_correct += correct
        total_seen += total

        accuracy = total_correct / total_seen
        batch_accuracies.append(accuracy)
        batch_numbers.append(batch_id)

        batch_id += 1

# -----------------------------
# Plot Graph
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(batch_numbers, batch_accuracies, marker='o')
plt.title("Model Accuracy Over Batches")
plt.xlabel("Batch Number")
plt.ylabel("Accuracy")
plt.grid(True)

# Show in VS Code
plt.show()

# Or save the graph:
# plt.savefig("accuracy_curve.png")