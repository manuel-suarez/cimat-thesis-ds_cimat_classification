import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader


class CimatDataset(Dataset):
    def __init__(
        self,
        base_dir,
        dataset,
        trainset,
        features_channels,
        features_extension,
        labels_extension,
        learning_dir,
        training_mode,
        max_images=None,
    ):
        super().__init__()
        # Initialization
        self.data_dir = os.path.join(
            base_dir,
            "data",
            "projects",
            "consorcio-ia",
            "data",
            f"oil_spills_{dataset}",
            "augmented_dataset",
        )
        self.features_dir = os.path.join(self.data_dir, "features")
        self.labels_dir = os.path.join(self.data_dir, "labels")
        self.csv_datadir = os.path.join(self.data_dir, "learningCSV", learning_dir)
        csv_dataset = pd.read_csv(
            os.path.join(
                self.csv_datadir,
                f"{training_mode}{trainset}.csv",
            ),
            nrows=max_images,
        )
        self.keys = csv_dataset["key"]
        channels = {"o": "ORIGIN", "v": "VAR", "w": "WIND"}
        self.features_channels = [channels[feat] for feat in features_channels]
        self.features_extension = features_extension
        self.labels_extension = labels_extension

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        # Load label
        y = torch.from_numpy(
            np.expand_dims(
                imread(
                    os.path.join(self.labels_dir, key + self.labels_extension),
                    as_gray=True,
                ).astype(np.float32)
                / 255.0,
                0,
            )
        )
        # Load features
        x = torch.stack(
            [
                torch.from_numpy(
                    imread(
                        os.path.join(
                            self.features_dir,
                            feature,
                            key + self.features_extension,
                        ),
                        as_gray=True,
                    )
                )
                for feature in self.features_channels
            ]
        )
        return x, y


def prepare_dataloaders(
    base_dir,
    dataset,
    trainset,
    feat_channels,
    max_train_images=None,
    max_valid_images=None,
    max_test_images=None,
):
    train_dataset = CimatDataset(
        base_dir=base_dir,
        dataset=dataset,
        trainset=trainset,
        features_channels=feat_channels,
        features_extension=".tiff",
        labels_extension=".pgm",
        learning_dir="trainingFiles",
        training_mode="train",
        max_images=max_train_images,
    )

    valid_dataset = CimatDataset(
        base_dir=base_dir,
        dataset=dataset,
        trainset=trainset,
        features_channels=feat_channels,
        features_extension=".tiff",
        labels_extension=".pgm",
        learning_dir="crossFiles",
        training_mode="cross",
        max_images=max_valid_images,
    )

    test_dataset = CimatDataset(
        base_dir=base_dir,
        dataset=dataset,
        trainset=trainset,
        features_channels=feat_channels,
        features_extension=".tiff",
        labels_extension=".pgm",
        learning_dir="testingFiles",
        training_mode="test",
        max_images=max_test_images,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=8, pin_memory=True, shuffle=True, num_workers=8
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=4, pin_memory=True, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=4, pin_memory=True, shuffle=False, num_workers=4
    )
    return train_dataloader, valid_dataloader, test_dataloader


def save_predictions(batch_idx, images, labels, predictions, directory):
    for idx_image, image in enumerate(images):
        image = np.concatenate((image[0, :, :], image[1, :, :], image[2, :, :]), axis=1)
        # print(f"\t{idx_image} image shape: ", image.shape)
        # image = np.transpose(image, (1, 2, 0))
        image = image * 255
        image = image.astype(np.uint8)
        image_p = Image.fromarray(image)
        image_p = image_p.convert("L")
        image_p.save(os.path.join(directory, f"batch{batch_idx}_image{idx_image}.png"))
    for idx_label, label in enumerate(labels):
        # label = np.transpose(label, (1, 2, 0))
        label = np.squeeze(label)
        # print(f"\t{idx_label} label shape: ", label.shape)
        label = label * 255
        label = label.astype(np.uint8)
        label_p = Image.fromarray(label)
        label_p = label_p.convert("L")
        label_p.save(os.path.join(directory, f"batch{batch_idx}_label{idx_label}.png"))
    for idx_prediction, prediction in enumerate(predictions):
        # prediction = np.transpose(prediction, (1, 2, 0))
        prediction = np.squeeze(prediction)
        # print(f"\t{idx_prediction} prediction shape: ", prediction.shape)
        prediction_p = Image.fromarray((prediction * 255).astype(np.uint8))
        prediction_p = prediction_p.convert("L")
        prediction_p.save(
            os.path.join(
                directory,
                f"batch{batch_idx}_prediction{idx_prediction}.png",
            )
        )


def save_figures(batch_idx, images, labels, predictions, directory):
    fig, axs = plt.subplots(1, 3, figsize=(12, 8))
    axs[0].imshow(images[0, 0, :, :])
    axs[0].set_title("Imagen")
    axs[1].imshow(predictions[0, 0, :, :])
    axs[1].set_title("Prediction")
    axs[2].imshow(labels[0, 0, :, :])
    axs[2].set_title("Label")
    plt.savefig(os.path.join(directory, f"result_batch{batch_idx}.png"))
    plt.close()


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as F

    home_dir = os.path.expanduser("~")
    feat_channels = "owv"

    train_dataset = CimatDataset(
        base_dir=home_dir,
        dataset="17",
        trainset="01",
        features_channels=feat_channels,
        features_extension=".tiff",
        labels_extension=".pgm",
        learning_dir="trainingFiles",
        training_mode="train",
    )
    image, label = train_dataset[0]
    print(f"Tensor image shape: {image.shape}")
    print(f"Tensor label shape: {label.shape}")
    np_image = image.numpy()
    np_label = label.numpy()
    print(f"Numpy image shape: {np_image.shape}")
    print(f"Numpy label shape: {np_label.shape}")

    print(
        f"Image: max: {np.max(np_image)}, min: {np.min(np_image)}, values: {np.unique(np_image)}"
    )
    print(
        f"Label: max: {np.max(np_label)}, min: {np.min(np_label)}, values: {np.unique(np_label)}"
    )

    fig, axs = plt.subplots(1, 2)
    pil_image = np.array(F.to_pil_image(image))
    pil_label = np.array(F.to_pil_image(label))
    axs[0].imshow(pil_image)
    axs[1].imshow(pil_label)
    plt.show()
