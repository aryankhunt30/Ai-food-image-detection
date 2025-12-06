from dataset import FoodH5Dataset
from model import FoodClassifier
import torch
import numpy as np

# Load trained model
model = FoodClassifier(num_classes=3)
model.load_state_dict(torch.load("model_food.pth", map_location="cpu"))
model.eval()

# Load dataset
dataset = FoodH5Dataset("/Users/aryankhunt/Downloads/Food detection/food_c101_n1000_r384x384x3.h5", 
                        transform=None)

correct = 0
total = 0

with torch.no_grad():
    for img, label in dataset:
        img = img.unsqueeze(0)  # add batch dimension
        out = model(img)
        _, pred = torch.max(out, 1)

        total += 1
        correct += int(pred.item() == label)

print(f"Model Accuracy: {correct / total * 100:.2f}%")