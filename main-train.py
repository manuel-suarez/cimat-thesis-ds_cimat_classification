import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import argparse

from utils import default
from models import build_model
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
    "--max_images",
    type=int,
    required=False,
    default=None,
    help="max images to load from training set",
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
model_name = "classification"
encoder_name = default(args.encoder_name, "base")
logging.info(f"Device: {device}")
logging.info(f"Encoder name: {encoder_name}")
logging.info(f"Epochs: {epochs}")
print("Device: ", device)
print("Encoder: ", encoder_name)
print("Epochs: ", epochs)

if __name__ == "__main__":
    # Prepare data loaders
    train_dataloader, valid_dataloader, test_dataloader = prepare_dataloaders(
        base_dir=os.path.join(base_path, "data", "cimat", "dataset-cimat"),
        max_images=args.max_images,
    )
    # Prepare model according to SLURM array task id
    # Output channels according to len of features channels len(oov)=3
    model = build_model(
        model_name="",
        encoder_name=encoder_name,
        in_channels=1,
        model_type="classification",
    ).to(device)
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
    trainer.fit([train_dataloader, valid_dataloader])
    # Create flag file to indicate main script that weight models has been generated
    f = open(os.path.join("outputs", encoder_name, model_name, "training.txt"), "x")
    f.close()
    logging.info(args.done_message)
    print(args.done_message)
