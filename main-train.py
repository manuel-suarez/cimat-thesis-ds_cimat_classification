import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import argparse

from utils import default
from models import select_model
from slurm import slurm_vars
from trainer import Trainer
from dataset import prepare_dataloaders

# Configure command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--epochs", type=int, required=True, default=5, help="number of epochs to train"
)
parser.add_argument(
    "--encoder_name", type=str, required=True, default="base", help="encoder to use"
)
parser.add_argument(
    "--dataset", type=str, required=True, default="17", help="num of dataset to use"
)
parser.add_argument(
    "--trainset",
    type=str,
    required=True,
    default="01",
    help="num of training-valid-test set to use",
)
parser.add_argument(
    "--feat_channels", type=str, required=True, default="oov", help="features to use"
)
parser.add_argument(
    "--weights_path",
    type=str,
    required=False,
    default="weights",
    help="path to store training weighs",
)
parser.add_argument(
    "--metrics_path",
    type=str,
    required=False,
    default="metrics",
    help="path to store metrics",
)
parser.add_argument(
    "--logging_path",
    type=str,
    required=False,
    default="outputs",
    help="path to store logging",
)
parser.add_argument(
    "--max_train_images",
    type=int,
    required=False,
    default=None,
    help="max images to load from training set",
)
parser.add_argument(
    "--max_valid_images",
    type=int,
    required=False,
    default=None,
    help="max images to load from validation set",
)
parser.add_argument(
    "--max_test_images",
    type=int,
    required=False,
    default=None,
    help="max images to load from test set",
)
parser.add_argument(
    "--done_message",
    type=str,
    required=False,
    default="Done!",
    help="message to show on training end",
)
args = parser.parse_args()
logging.info("Args: ", args)
print("Args: ", args)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(
        args.logging_path,
        f"slurm-training_{slurm_vars['array_job_id']}_{slurm_vars['array_task_id']}.out",
    ),
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)

# Initial configuration
epochs = args.epochs
base_path = os.path.expanduser("~")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model selection is based on array_task_id slurm job id
model_name, model_arch = select_model(slurm_vars["array_task_id"] - 1)
# We are using "base" as encoder name because we are not implemented the encoder library module
encoder_name = default(args.encoder_name, "base")
dataset = default(args.dataset, "17")
trainset = default(args.trainset, "01")
feat_channels = default(args.feat_channels, "oov")
logging.info(f"Device: {device}")
logging.info(f"Model name: {model_name}")
logging.info(f"Encoder name: {encoder_name}")
logging.info(f"Epochs: {epochs}")
logging.info(f"Dataset: {dataset}")
logging.info(f"Trainset: {trainset}")
logging.info(f"Features channels: {feat_channels}")
print("Device: ", device)
print("Model: ", model_name)
print("Encoder: ", encoder_name)
print("Epochs: ", epochs)
print("Dataset: ", dataset)
print("Trainset: ", trainset)
print("Features channels: ", feat_channels)

if __name__ == "__main__":
    # Prepare data loaders
    train_dataloader, valid_dataloader, test_dataloader = prepare_dataloaders(
        base_dir=base_path,
        dataset=dataset,
        trainset=trainset,
        feat_channels=feat_channels,
        max_train_images=args.max_train_images,
        max_valid_images=args.max_valid_images,
        max_test_images=args.max_test_images,
    )
    # Prepare model according to SLURM array task id
    # Output channels according to len of features channels len(oov)=3
    model = model_arch(in_channels=len(feat_channels)).to(device)
    # Prepare optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    # Configure paths
    metrics_path = os.path.join(args.metrics_path, encoder_name, model_name)
    weights_path = os.path.join(args.weights_path, encoder_name, model_name)
    os.makedirs(metrics_path, exist_ok=True)
    os.makedirs(weights_path, exist_ok=True)

    # Instance trainer
    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        epochs,
        device,
        metrics_path=metrics_path,
        weights_path=weights_path,
    )
    # Start training process
    trainer.fit([train_dataloader, valid_dataloader], dataset, trainset, feat_channels)
    logging.info(args.done_message)
    print(args.done_message)
