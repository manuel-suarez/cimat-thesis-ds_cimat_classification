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
            "precision": 0.0,
            "specificity": 0.0,
            "recall": 0.0,
        }
        self.model.train()
        for image, label in dataloader:
            image, label = image.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            # Get predictions
            output = self.model(image)
            # Squeeze dimension for binary classification
            output = output.squeeze(dim=1)
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
            "precision": 0.0,
            "specificity": 0.0,
            "recall": 0.0,
        }
        self.model.eval()
        for image, label in dataloader:
            image, label = image.to(self.device), label.to(self.device)

            with torch.no_grad():
                output = self.model(image)
                output = output.squeeze(dim=1)
                loss = self.loss_fn(output, label)
                valid_metrics = calculate_metrics(output, label)

                valid_loss += loss.item()

        valid_loss = valid_loss / len(dataloader)
        return valid_loss, valid_metrics

    def fit(self, dataloaders):
        train_dataloader, valid_dataloader = dataloaders
        # Prepare metrics dict to save to a dataframe
        metrics = {
            "epoch": [],
            "train_loss": [],
            "train_accuracy": [],
            "train_precision": [],
            "train_specificity": [],
            "train_recall": [],
            "valid_loss": [],
            "valid_accuracy": [],
            "valid_precision": [],
            "valid_specificity": [],
            "valid_recall": [],
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
                f"Epoch {epoch + 1}/{self.epochs}, train loss: {train_loss}, train accuracy: {train_metrics['accuracy']}, train precision: {train_metrics['precision']},train specificity: {train_metrics['specificity']}, train recall: {train_metrics['recall']}, valid loss: {valid_loss}, valid accuracy: {valid_metrics['accuracy']}, valid precision: {valid_metrics['precision']} valid specificity: {valid_metrics['specificity']}, valid recall: {valid_metrics['recall']}"
            )
            print(
                f"Epoch {epoch + 1}/{self.epochs}, train loss: {train_loss}, train accuracy: {train_metrics['accuracy']}, train precision: {train_metrics['precision']},train specificity: {train_metrics['specificity']}, train recall: {train_metrics['recall']}, valid loss: {valid_loss}, valid accuracy: {valid_metrics['accuracy']}, valid precision: {valid_metrics['precision']} valid specificity: {valid_metrics['specificity']}, valid recall: {valid_metrics['recall']}"
            )
            metrics["epoch"].append(epoch + 1)
            metrics["train_loss"].append(train_loss)
            metrics["train_accuracy"].append(train_metrics["accuracy"])
            metrics["train_precision"].append(train_metrics["precision"])
            metrics["train_specificity"].append(train_metrics["specificity"])
            metrics["train_recall"].append(train_metrics["recall"])
            metrics["valid_loss"].append(valid_loss)
            metrics["valid_accuracy"].append(valid_metrics["accuracy"])
            metrics["valid_precision"].append(valid_metrics["precision"])
            metrics["valid_specificity"].append(valid_metrics["specificity"])
            metrics["valid_recall"].append(valid_metrics["recall"])
            metrics["start_time"].append(start_time)
            metrics["train_time"].append(train_time)
            metrics["valid_time"].append(valid_time)

            # Save weights every 20 epochs
            if (epoch + 1) % 20 == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.weights_path,
                        f"weights_{epoch+1}_epochs.pth",
                    ),
                )
            # Save metrics to CSV on each epoch
            metrics_df = pd.DataFrame.from_dict(metrics)
            metrics_df.to_csv(
                os.path.join(
                    self.metrics_path,
                    "metrics.csv",
                )
            )

        endTime = time.time()
        logging.info("--== Training end ==--")
        logging.info(
            "Total time training and validation: {:.2f}s".format(endTime - startTime)
        )
        print("--== Training end ==--")
        print("Total time training and validation: {:.2f}s".format(endTime - startTime))
