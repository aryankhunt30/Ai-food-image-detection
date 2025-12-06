import torch
import cv2
import numpy as np
from model import load_model
from torchvision import transforms


def predict(image_path, model_path="model_food.pth"):
    model = load_model(model_path, num_classes=3)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (300, 300))

    transform = transforms.ToTensor()
    input_tensor = transform(image_resized).unsqueeze(0)

    with torch.no_grad():
        preds = model(input_tensor)
        probs = torch.softmax(preds, dim=1).numpy()[0]

    classes = ["real", "ai", "edited"]
    return {cls: float(prob) for cls, prob in zip(classes, probs)}