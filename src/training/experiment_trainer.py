from pathlib import Path
import json
import time

import torch
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


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
        scheduler=None,
        loss_name="cross_entropy",
        class_names=None,
        config=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = device
        self.epochs = epochs
        self.run_name = run_name
        self.loss_name = loss_name
        self.class_names = class_names
        self.config = config

        # ----------------------------------------------------
        # Output directory for this experiment
        # ----------------------------------------------------

        self.run_dir = Path(output_dir) / run_name
        self.run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Training state
        # ----------------------------------------------------

        self.history = []

        self.best_val_loss = float("inf")
        self.best_epoch = 0

    # ========================================================
    # Target preparation
    # ========================================================

    def prepare_targets(
        self,
        labels,
        num_classes,
    ):

        if self.loss_name == "cross_entropy":

            return labels

        if self.loss_name == "bce":

            return torch.nn.functional.one_hot(
                labels,
                num_classes=num_classes,
            ).float()

        raise ValueError(
            f"Unsupported loss: {self.loss_name}"
        )

    # ========================================================
    # Train one epoch
    # ========================================================

    def train_one_epoch(self):

        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            # -----------------------------------------------
            # Clear gradients
            # -----------------------------------------------

            self.optimizer.zero_grad(
                set_to_none=True
            )

            # -----------------------------------------------
            # Forward
            # -----------------------------------------------

            outputs = self.model(images)

            # -----------------------------------------------
            # Prepare targets
            # -----------------------------------------------

            targets = self.prepare_targets(
                labels,
                outputs.shape[1],
            )

            # -----------------------------------------------
            # Loss
            # -----------------------------------------------

            loss = self.criterion(
                outputs,
                targets,
            )

            # -----------------------------------------------
            # Backward
            # -----------------------------------------------

            loss.backward()

            # -----------------------------------------------
            # Update weights
            # -----------------------------------------------

            self.optimizer.step()

            # -----------------------------------------------
            # Statistics
            # -----------------------------------------------

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

    # ========================================================
    # Validation
    # ========================================================

    @torch.no_grad()
    def validate_one_epoch(self):

        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        all_predictions = []
        all_labels = []

        for images, labels in self.validation_loader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            # -----------------------------------------------
            # Forward
            # -----------------------------------------------

            outputs = self.model(images)

            # -----------------------------------------------
            # Prepare targets
            # -----------------------------------------------

            targets = self.prepare_targets(
                labels,
                outputs.shape[1],
            )

            # -----------------------------------------------
            # Loss
            # -----------------------------------------------

            loss = self.criterion(
                outputs,
                targets,
            )

            # -----------------------------------------------
            # Statistics
            # -----------------------------------------------

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

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

        average_loss = total_loss / total
        accuracy = correct / total

        metrics = self.calculate_metrics(
            all_labels,
            all_predictions,
        )

        return (
            average_loss,
            accuracy,
            metrics,
        )

    # ========================================================
    # Metrics
    # ========================================================

    def calculate_metrics(
        self,
        labels,
        predictions,
    ):

        macro_precision = precision_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )

        macro_recall = recall_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )

        macro_f1 = f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )

        class_precision = precision_score(
            labels,
            predictions,
            average=None,
            zero_division=0,
        )

        class_recall = recall_score(
            labels,
            predictions,
            average=None,
            zero_division=0,
        )

        class_f1 = f1_score(
            labels,
            predictions,
            average=None,
            zero_division=0,
        )

        metrics = {
            "macro_precision": float(
                macro_precision
            ),
            "macro_recall": float(
                macro_recall
            ),
            "macro_f1": float(
                macro_f1
            ),
            "per_class": {},
        }

        for index in range(
            len(class_precision)
        ):

            if self.class_names is not None:

                class_name = self.class_names[
                    index
                ]

            else:

                class_name = str(index)

            metrics["per_class"][
                class_name
            ] = {

                "precision": float(
                    class_precision[index]
                ),

                "recall": float(
                    class_recall[index]
                ),

                "f1": float(
                    class_f1[index]
                ),
            }

        # ----------------------------------------------------
        # Weakest class by precision
        # ----------------------------------------------------

        lowest_precision_class = min(
            metrics["per_class"],
            key=lambda name:
                metrics["per_class"][name][
                    "precision"
                ],
        )

        # ----------------------------------------------------
        # Weakest class by recall
        # ----------------------------------------------------

        lowest_recall_class = min(
            metrics["per_class"],
            key=lambda name:
                metrics["per_class"][name][
                    "recall"
                ],
        )

        metrics[
            "lowest_precision_class"
        ] = lowest_precision_class

        metrics[
            "lowest_recall_class"
        ] = lowest_recall_class

        return metrics

    # ========================================================
    # Learning rate
    # ========================================================

    def get_learning_rate(self):

        return self.optimizer.param_groups[0][
            "lr"
        ]

    # ========================================================
    # JSON
    # ========================================================

    def save_json(
        self,
        filename,
        data,
    ):

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

    # ========================================================
    # Checkpoint
    # ========================================================

    def save_checkpoint(
        self,
        filename,
        epoch,
        train_loss,
        train_accuracy,
        val_loss,
        val_accuracy,
        val_metrics,
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

            "val_metrics":
                val_metrics,
        }

        if self.scheduler is not None:

            checkpoint[
                "scheduler_state_dict"
            ] = self.scheduler.state_dict()

        torch.save(
            checkpoint,
            self.run_dir / filename,
        )

    # ========================================================
    # Config
    # ========================================================

    def save_config(self):

        if self.config is None:
            return

        self.save_json(
            "config.json",
            self.config,
        )

    # ========================================================
    # History
    # ========================================================

    def save_history(self):

        self.save_json(
            "history.json",
            self.history,
        )

    # ========================================================
    # Best model
    # ========================================================

    def save_best_model(
        self,
        epoch,
        train_loss,
        train_accuracy,
        val_loss,
        val_accuracy,
        val_metrics,
    ):

        self.save_checkpoint(

            filename="best_model.pt",

            epoch=epoch,

            train_loss=train_loss,

            train_accuracy=train_accuracy,

            val_loss=val_loss,

            val_accuracy=val_accuracy,

            val_metrics=val_metrics,
        )

    # ========================================================
    # Scheduler
    # ========================================================

    def step_scheduler(
        self,
        val_loss,
    ):

        if self.scheduler is None:
            return

        scheduler_name = (

            self.config
            .get("scheduler", {})
            .get("name", "none")

            if self.config is not None

            else "none"
        )

        if scheduler_name == "reduce_on_plateau":

            self.scheduler.step(
                val_loss
            )

        else:

            self.scheduler.step()

    # ========================================================
    # Build final metrics
    # ========================================================

    def build_metrics(self):

        best_result = min(
            self.history,
            key=lambda x: x["val_loss"],
        )

        final_result = self.history[-1]

        return {

            "run_name":
                self.run_name,

            "best_epoch":
                self.best_epoch,

            "best_val_loss":
                best_result["val_loss"],

            "best_val_accuracy":
                best_result["val_accuracy"],

            "best_macro_precision":
                best_result["macro_precision"],

            "best_macro_recall":
                best_result["macro_recall"],

            "best_macro_f1":
                best_result["macro_f1"],

            "lowest_recall_class":
                best_result[
                    "lowest_recall_class"
                ],

            "lowest_precision_class":
                best_result[
                    "lowest_precision_class"
                ],

            "final_train_loss":
                final_result["train_loss"],

            "final_train_accuracy":
                final_result[
                    "train_accuracy"
                ],

            "final_val_loss":
                final_result["val_loss"],

            "final_val_accuracy":
                final_result[
                    "val_accuracy"
                ],

            "generalization_gap":
                (
                    final_result[
                        "train_accuracy"
                    ]
                    -
                    final_result[
                        "val_accuracy"
                    ]
                ),
        }

    # ========================================================
    # Training
    # ========================================================

    def train(self):

        self.save_config()

        for epoch in range(
            1,
            self.epochs + 1,
        ):

            start_time = time.perf_counter()

            # ------------------------------------------------
            # Train
            # ------------------------------------------------

            train_loss, train_accuracy = (
                self.train_one_epoch()
            )

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            (
                val_loss,
                val_accuracy,
                val_metrics,
            ) = self.validate_one_epoch()

            # ------------------------------------------------
            # IMPORTANT:
            # Capture LR BEFORE scheduler step.
            #
            # This is the LR that was actually used
            # during this epoch.
            # ------------------------------------------------

            learning_rate = (
                self.get_learning_rate()
            )

            # ------------------------------------------------
            # Scheduler
            # ------------------------------------------------

            self.step_scheduler(
                val_loss
            )

            # ------------------------------------------------
            # Timing
            # ------------------------------------------------

            epoch_time = (
                time.perf_counter()
                - start_time
            )

            # ------------------------------------------------
            # Epoch result
            # ------------------------------------------------

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

                "macro_precision":
                    val_metrics[
                        "macro_precision"
                    ],

                "macro_recall":
                    val_metrics[
                        "macro_recall"
                    ],

                "macro_f1":
                    val_metrics[
                        "macro_f1"
                    ],

                "lowest_precision_class":
                    val_metrics[
                        "lowest_precision_class"
                    ],

                "lowest_recall_class":
                    val_metrics[
                        "lowest_recall_class"
                    ],

                "learning_rate":
                    learning_rate,

                "epoch_time":
                    epoch_time,
            }

            self.history.append(
                epoch_result
            )

            # ------------------------------------------------
            # Console log
            # ------------------------------------------------

            print(

                f"Epoch "
                f"[{epoch:02d}/{self.epochs}] | "

                f"Train Loss: "
                f"{train_loss:.4f} | "

                f"Train Acc: "
                f"{train_accuracy:.4f} | "

                f"Val Loss: "
                f"{val_loss:.4f} | "

                f"Val Acc: "
                f"{val_accuracy:.4f} | "

                f"Macro F1: "
                f"{val_metrics['macro_f1']:.4f} | "

                f"LR: "
                f"{learning_rate:.2e} | "

                f"Time: "
                f"{epoch_time:.1f}s"
            )

            # ------------------------------------------------
            # Save history
            # ------------------------------------------------

            self.save_history()

            # ------------------------------------------------
            # Save latest checkpoint
            # ------------------------------------------------

            self.save_checkpoint(

                filename="checkpoint_last.pt",

                epoch=epoch,

                train_loss=train_loss,

                train_accuracy=train_accuracy,

                val_loss=val_loss,

                val_accuracy=val_accuracy,

                val_metrics=val_metrics,
            )

            # ------------------------------------------------
            # Best model
            #
            # Best = lowest validation loss
            # ------------------------------------------------

            if val_loss < self.best_val_loss:

                self.best_val_loss = val_loss

                self.best_epoch = epoch

                self.save_best_model(

                    epoch=epoch,

                    train_loss=train_loss,

                    train_accuracy=train_accuracy,

                    val_loss=val_loss,

                    val_accuracy=val_accuracy,

                    val_metrics=val_metrics,
                )

                print(
                    "  -> New best model saved."
                )

        # ====================================================
        # Final metrics
        # ====================================================

        metrics = self.build_metrics()

        self.save_json(
            "metrics.json",
            metrics,
        )

        print("\nRun completed:")

        print(
            f"Best epoch: "
            f"{self.best_epoch}"
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
            f"Best Macro F1: "
            f"{metrics['best_macro_f1']:.4f}"
        )

        print(
            f"Generalization Gap: "
            f"{metrics['generalization_gap']:.4f}"
        )

        return self.history, metrics