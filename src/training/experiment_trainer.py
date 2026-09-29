from pathlib import Path
import json
import time

import torch


class ExperimentTrainer:

    def __init__(
        self,
        model,
        train_loader,
        validation_loader,
        criterion,
        optimizer,
        device,
        epochs,
        run_name,
        output_dir,
        config=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.run_name = run_name
        self.config = config

        self.run_dir = (
            Path(output_dir) / run_name
        )

        self.run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.history = []

        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def train_one_epoch(self):
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad(
                set_to_none=True
            )

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                labels,
            )

            loss.backward()

            self.optimizer.step()

            batch_size = labels.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += batch_size

        average_loss = total_loss / total
        accuracy = correct / total

        return average_loss, accuracy

    @torch.no_grad()
    def validate_one_epoch(self):
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.validation_loader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                labels,
            )

            batch_size = labels.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += batch_size

        average_loss = total_loss / total
        accuracy = correct / total

        return average_loss, accuracy

    def save_json(self, filename, data):

        path = self.run_dir / filename

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                indent=4,
            )

    def save_checkpoint(
        self,
        filename,
        epoch,
        train_loss,
        train_accuracy,
        val_loss,
        val_accuracy,
    ):

        checkpoint = {
            "epoch": epoch,

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "train_loss":
                train_loss,

            "train_accuracy":
                train_accuracy,

            "val_loss":
                val_loss,

            "val_accuracy":
                val_accuracy,
        }

        torch.save(
            checkpoint,
            self.run_dir / filename,
        )

    def save_config(self):

        if self.config is None:
            return

        self.save_json(
            "config.json",
            self.config,
        )

    def save_history(self):

        self.save_json(
            "history.json",
            self.history,
        )

    def save_best_model(
        self,
        epoch,
        train_loss,
        train_accuracy,
        val_loss,
        val_accuracy,
    ):

        self.save_checkpoint(
            filename="best_model.pt",
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
        )

    def build_metrics(self):

        best_result = min(
            self.history,
            key=lambda x: x["val_loss"],
        )

        final_result = self.history[-1]

        metrics = {
            "run_name":
                self.run_name,

            "best_epoch":
                self.best_epoch,

            "best_val_loss":
                best_result["val_loss"],

            "best_val_accuracy":
                best_result["val_accuracy"],

            "final_train_loss":
                final_result["train_loss"],

            "final_train_accuracy":
                final_result["train_accuracy"],

            "final_val_loss":
                final_result["val_loss"],

            "final_val_accuracy":
                final_result["val_accuracy"],
        }

        metrics["generalization_gap"] = (
            final_result["train_accuracy"]
            - final_result["val_accuracy"]
        )

        return metrics

    def train(self):

        self.save_config()

        for epoch in range(
            1,
            self.epochs + 1,
        ):

            start_time = time.perf_counter()

            train_loss, train_accuracy = (
                self.train_one_epoch()
            )

            val_loss, val_accuracy = (
                self.validate_one_epoch()
            )

            epoch_time = (
                time.perf_counter()
                - start_time
            )

            epoch_result = {
                "epoch": epoch,

                "train_loss":
                    train_loss,

                "train_accuracy":
                    train_accuracy,

                "val_loss":
                    val_loss,

                "val_accuracy":
                    val_accuracy,

                "epoch_time":
                    epoch_time,
            }

            self.history.append(
                epoch_result
            )

            print(
                f"Epoch [{epoch:02d}/{self.epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Save every epoch
            self.save_history()

            self.save_checkpoint(
                filename="checkpoint_last.pt",
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
            )

            # Best model
            if val_loss < self.best_val_loss:

                self.best_val_loss = val_loss
                self.best_epoch = epoch

                self.save_best_model(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_accuracy=train_accuracy,
                    val_loss=val_loss,
                    val_accuracy=val_accuracy,
                )

                print(
                    "  -> New best model saved."
                )

        metrics = self.build_metrics()

        self.save_json(
            "metrics.json",
            metrics,
        )

        print("\nRun completed:")
        print(
            f"Best epoch: {self.best_epoch}"
        )
        print(
            f"Best Val Loss: "
            f"{metrics['best_val_loss']:.4f}"
        )
        print(
            f"Best Val Accuracy: "
            f"{metrics['best_val_accuracy']:.4f}"
        )
        print(
            f"Generalization Gap: "
            f"{metrics['generalization_gap']:.4f}"
        )

        return self.history, metrics