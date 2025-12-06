import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


class FoodClassifier(nn.Module):
    """
    EfficientNet-B3 based classifier for detecting:
    1. Real food images
    2. AI-generated food images
    3. AI-edited food images
    """

    def __init__(self, num_classes=3):
        super().__init__()

        # Load pretrained EfficientNet-B3 weights
        self.base = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)

        # Replace final classification layer
        in_features = self.base.classifier[1].in_features
        self.base.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base(x)


def load_model(model_path, num_classes=3, device="cpu"):
    """
    Loads a trained model checkpoint.

    Args:
        model_path: path to .pth model file
        num_classes: number of output classes
        device: "cpu" or "cuda"

    Returns:
        Loaded model in eval mode.
    """
    model = FoodClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model