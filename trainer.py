import os
import time
import torch
import logging
import pandas as pd

from metrics import calculate_metrics


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        epochs,
        device,
        metrics_path,
        weights_path,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.device = device
        self.metrics_path = metrics_path
        self.weights_path = weights_path

    def train_step(self, dataloader):
        train_loss = 0.0
        train_metrics = {
            "accuracy": 0.0,
            "specificity": 0.0,
            "sensitivity": 0.0,
            "dice": 0.0,
            "iou": 0.0,
        }
        self.model.train()
        for image, label in dataloader:
            image, label = image.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            # Get predictions
            output = self.model(image)
            # Calculate loss and metrics
            loss = self.loss_fn(output, label)
            train_metrics = calculate_metrics(output, label)
            # Update
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / len(dataloader)
        return train_loss, train_metrics

    def valid_step(self, dataloader):
        valid_loss = 0.0
        valid_metrics = {
            "accuracy": 0.0,
            "specificity": 0.0,
            "sensitivity": 0.0,
            "dice": 0.0,
            "iou": 0.0,
        }
        self.model.eval()
        for image, label in dataloader:
            image, label = image.to(self.device), label.to(self.device)

            with torch.no_grad():
                output = self.model(image)
                loss = self.loss_fn(output, label)
                valid_metrics = calculate_metrics(output, label)

                valid_loss += loss.item()

        valid_loss = valid_loss / len(dataloader)
        return valid_loss, valid_metrics

    def fit(self, dataloaders, dataset, trainset, feat_channels):
        train_dataloader, valid_dataloader = dataloaders
        # Prepare metrics dict to save to a dataframe
        metrics = {
            "epoch": [],
            "train_loss": [],
            "train_accuracy": [],
            "train_specificity": [],
            "train_sensitivity": [],
            "train_dice": [],
            "train_iou": [],
            "valid_loss": [],
            "valid_accuracy": [],
            "valid_specificity": [],
            "valid_sensitivity": [],
            "valid_dice": [],
            "valid_iou": [],
            "start_time": [],
            "train_time": [],
            "valid_time": [],
        }

        logging.info("--== Training start ==--")
        print("--== Training start ==--")
        startTime = time.time()

        for epoch in range(self.epochs):
            start_time = time.time()
            train_loss, train_metrics = self.train_step(train_dataloader)
            train_time = time.time()
            valid_loss, valid_metrics = self.valid_step(valid_dataloader)
            valid_time = time.time()
            logging.info(
                f"Epoch {epoch + 1}/{self.epochs}, train loss: {train_loss}, train accuracy: {train_metrics['accuracy']}, train specificity: {train_metrics['specificity']}, train sensitivity: {train_metrics['sensitivity']}, train dice score: {train_metrics['dice']}, train IoU: {train_metrics['iou']}, valid loss: {valid_loss}, valid accuracy: {valid_metrics['accuracy']}, valid specificity: {valid_metrics['specificity']}, valid sensitivity: {valid_metrics['sensitivity']}, valid dice: {valid_metrics['dice']}, valid iou: {valid_metrics['iou']}"
            )
            print(
                f"Epoch {epoch + 1}/{self.epochs}, train loss: {train_loss}, train accuracy: {train_metrics['accuracy']}, train specificity: {train_metrics['specificity']}, train sensitivity: {train_metrics['sensitivity']}, train dice score: {train_metrics['dice']}, train IoU: {train_metrics['iou']}, valid loss: {valid_loss}, valid accuracy: {valid_metrics['accuracy']}, valid specificity: {valid_metrics['specificity']}, valid sensitivity: {valid_metrics['sensitivity']}, valid dice: {valid_metrics['dice']}, valid iou: {valid_metrics['iou']}"
            )
            metrics["epoch"].append(epoch + 1)
            metrics["train_loss"].append(train_loss)
            metrics["train_accuracy"].append(train_metrics["accuracy"])
            metrics["train_specificity"].append(train_metrics["specificity"])
            metrics["train_sensitivity"].append(train_metrics["sensitivity"])
            metrics["train_dice"].append(train_metrics["dice"])
            metrics["train_iou"].append(train_metrics["iou"])
            metrics["valid_loss"].append(valid_loss)
            metrics["valid_accuracy"].append(valid_metrics["accuracy"])
            metrics["valid_specificity"].append(valid_metrics["specificity"])
            metrics["valid_sensitivity"].append(valid_metrics["sensitivity"])
            metrics["valid_dice"].append(valid_metrics["dice"])
            metrics["valid_iou"].append(valid_metrics["iou"])
            metrics["start_time"].append(start_time)
            metrics["train_time"].append(train_time)
            metrics["valid_time"].append(valid_time)

            # Save weights every 20 epochs
            if (epoch + 1) % 20 == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.weights_path,
                        f"weights_{dataset}_{trainset}_{feat_channels}_{epoch+1}_epochs.pth",
                    ),
                )
            # Save metrics to CSV on each epoch
            metrics_df = pd.DataFrame.from_dict(metrics)
            metrics_df.to_csv(
                os.path.join(
                    self.metrics_path,
                    "metrics_{dataset}_{trainset}_{feat_channels}.csv",
                )
            )

        endTime = time.time()
        logging.info("--== Training end ==--")
        logging.info(
            "Total time training and validation: {:.2f}s".format(endTime - startTime)
        )
        print("--== Training end ==--")
        print("Total time training and validation: {:.2f}s".format(endTime - startTime))
