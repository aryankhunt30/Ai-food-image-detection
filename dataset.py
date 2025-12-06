import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FoodH5Dataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform

        # Open the H5 file
        self.h5_file = h5py.File(self.h5_path, "r")

        # Load datasets
        self.images = self.h5_file["images"]
        self.labels = self.h5_file["category"]
        self.class_names = [
            name.decode("utf-8") for name in self.h5_file["category_names"]
        ]

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        # Load image
        img = self.images[index].astype(np.float32) / 255.0

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]
        else:
            img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)

        # -------- FIXED UNIVERSAL LABEL EXTRACTION --------
        raw_label = self.labels[index]
        raw_label = np.array(raw_label)

        if raw_label.size == 1:
            label = int(raw_label.reshape(-1)[0])
        else:
            # If label array contains multiple values (rare), take the first element
            label = int(raw_label.reshape(-1)[0])
        # ---------------------------------------------------

        return img, label

    def get_class_names(self):
        return self.class_names


def create_dataloaders(
    h5_path,
    train_transform=None,
    val_transform=None,
    batch_size=32,
    val_split=0.2,
):
    dataset = FoodH5Dataset(h5_path, transform=None)

    n = len(dataset)
    val_size = int(n * val_split)
    train_size = n - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader