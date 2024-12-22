import os
import torch
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader, random_split


class CimatDataset(Dataset):
    def __init__(
        self,
        base_dir,
        return_names=False,
        max_images=None,
    ):
        super().__init__()
        # Initialization
        data_dir = os.path.join(
            base_dir,
            "classification",
        )
        self.return_names = return_names
        # Open oil-not oil directories
        oil_dir = os.path.join(data_dir, "oil")
        not_oil_dir = os.path.join(data_dir, "not_oil")

        # Images
        oil_images = [os.path.join(oil_dir, fname) for fname in os.listdir(oil_dir)]
        not_oil_images = [
            os.path.join(not_oil_dir, fname) for fname in os.listdir(not_oil_dir)
        ]
        self.images = oil_images + not_oil_images
        random.shuffle(self.images)
        # Labels
        oil_labels = {os.path.join(oil_dir, fname): 1 for fname in os.listdir(oil_dir)}
        not_oil_labels = {
            os.path.join(not_oil_dir, fname): 0 for fname in os.listdir(not_oil_dir)
        }
        self.labels = dict(oil_labels, **not_oil_labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        # Load image
        image = torch.from_numpy(
            np.expand_dims(
                imread(
                    image_name,
                    as_gray=True,
                ).astype(np.float32),
                0,
            )
        )
        if self.return_names:
            return image, self.labels[image_name], image_name
        else:
            return image, self.labels[image_name]


def prepare_dataloaders(
    base_dir,
    train_batch_size=8,
    valid_batch_size=4,
    test_batch_size=4,
    return_names=False,
    max_images=None,
):
    base_dataset = CimatDataset(
        base_dir=base_dir,
        return_names=return_names,
        max_images=max_images,
    )
    train_dataset, valid_dataset, test_dataset = random_split(
        base_dataset, [0.7, 0.2, 0.1]
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=8,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=4,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=4,
    )
    return train_dataloader, valid_dataloader, test_dataloader


if __name__ == "__main__":
    import numpy as np

    home_dir = os.path.expanduser("~")
    data_dir = os.path.join(home_dir, "data", "cimat", "dataset-cimat")

    train_dataset = CimatDataset(
        base_dir=data_dir,
    )
    print(f"Dataset len: {len(train_dataset)}")
    image, label = train_dataset[0]
    print(f"Tensor image shape: {image.shape}")
    print(f"Tensor label: {label}")

    train_dataloader, valid_dataloader, test_dataloader = prepare_dataloaders(data_dir)
    print(f"Batches train: {len(train_dataloader)}")
    print(f"Batches valid: {len(valid_dataloader)}")
    print(f"Batches test: {len(test_dataloader)}")

    images, labels = next(iter(train_dataloader))
    print(f"Len images, labels: {len(images)}, {len(labels)}")
    for image, label in zip(images, labels):
        print(f"Tensor shape, label: {image.shape}, {label}")
