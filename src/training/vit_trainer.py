from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.optim import AdamW

from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingLR,
)

from src.evaluation.evaluator import Evaluator


class ViTTrainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        experiment_config,
        device,
        class_names,
        output_dir,
    ):

        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.experiment_config = experiment_config

        self.device = device
        self.class_names = class_names

        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ====================================================
        # Training configuration
        # ====================================================

        self.epochs = experiment_config[
            "training"
        ]["epochs"]

        self.strategy = model.strategy

        # ====================================================
        # Loss
        # ====================================================

        self.criterion = nn.CrossEntropyLoss()

        # ====================================================
        # Optimizer
        # ====================================================

        optimizer_config = experiment_config[
            "optimizer"
        ]

        optimizer_name = optimizer_config.get(
            "name",
            "AdamW",
        )

        if optimizer_name != "AdamW":

            raise ValueError(
                f"Unsupported optimizer: "
                f"{optimizer_name}"
            )

        self.optimizer = AdamW(
            model.get_trainable_parameters(),
            lr=optimizer_config["lr"],
            weight_decay=optimizer_config.get(
                "weight_decay",
                0.0,
            ),
        )

        # ====================================================
        # Scheduler
        # ====================================================

        scheduler_config = experiment_config[
            "scheduler"
        ]

        scheduler_name = scheduler_config.get(
            "name"
        )

        if scheduler_name == "ReduceLROnPlateau":

            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=scheduler_config.get(
                    "mode",
                    "min",
                ),
                factor=scheduler_config.get(
                    "factor",
                    0.1,
                ),
                patience=scheduler_config.get(
                    "patience",
                    3,
                ),
            )

        elif scheduler_name == "CosineAnnealingLR":

            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
            )

        else:

            raise ValueError(
                f"Unsupported scheduler: "
                f"{scheduler_name}"
            )

        self.scheduler_name = scheduler_name

        # ====================================================
        # Evaluator
        # ====================================================

        self.evaluator = Evaluator()

        # ====================================================
        # Training state
        # ====================================================

        self.best_val_macro_f1 = -float("inf")

        self.best_val_loss = float("inf")

        self.history = []

    # ========================================================
    # Training mode
    # ========================================================

    def _set_training_mode(self):

        if self.strategy == "feature_extraction":

            self.model.model.eval()

            self.model.model.heads.head.train()

        elif self.strategy in [
            "fine_tune_stage_1",
            "fine_tune_stage_2",
            "from_scratch",
        ]:

            self.model.train()

        else:

            raise ValueError(
                f"Unsupported ViT strategy: "
                f"{self.strategy}"
            )

    # ========================================================
    # Train one epoch
    # ========================================================

    def train_epoch(self):

        self._set_training_mode()

        running_loss = 0.0

        all_targets = []
        all_predictions = []
        all_probabilities = []

        for images, targets in self.train_loader:

            images = images.to(
                self.device
            )

            targets = targets.to(
                self.device
            )

            self.optimizer.zero_grad()

            outputs = self.model(
                images
            )

            loss = self.criterion(
                outputs,
                targets,
            )

            loss.backward()

            self.optimizer.step()

            running_loss += (
                loss.item()
                * images.size(0)
            )

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            predictions = probabilities.argmax(
                dim=1
            )

            all_targets.append(
                targets.detach()
                .cpu()
                .numpy()
            )

            all_predictions.append(
                predictions.detach()
                .cpu()
                .numpy()
            )

            all_probabilities.append(
                probabilities.detach()
                .cpu()
                .numpy()
            )

        epoch_loss = (
            running_loss
            / len(
                self.train_loader.dataset
            )
        )

        y_true = np.concatenate(
            all_targets
        )

        y_pred = np.concatenate(
            all_predictions
        )

        y_prob = np.concatenate(
            all_probabilities
        )

        metrics = self.evaluator.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
        )

        return epoch_loss, metrics

    # ========================================================
    # Validation
    # ========================================================

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = 0.0

        all_targets = []
        all_predictions = []
        all_probabilities = []

        for images, targets in self.val_loader:

            images = images.to(
                self.device
            )

            targets = targets.to(
                self.device
            )

            outputs = self.model(
                images
            )

            loss = self.criterion(
                outputs,
                targets,
            )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            predictions = probabilities.argmax(
                dim=1
            )

            all_targets.append(
                targets.cpu()
                .numpy()
            )

            all_predictions.append(
                predictions.cpu()
                .numpy()
            )

            all_probabilities.append(
                probabilities.cpu()
                .numpy()
            )

        epoch_loss = (
            running_loss
            / len(
                self.val_loader.dataset
            )
        )

        y_true = np.concatenate(
            all_targets
        )

        y_pred = np.concatenate(
            all_predictions
        )

        y_prob = np.concatenate(
            all_probabilities
        )

        metrics = self.evaluator.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
        )

        return epoch_loss, metrics

    # ========================================================
    # Scheduler
    # ========================================================

    def _scheduler_step(
        self,
        val_loss,
    ):

        if (
            self.scheduler_name
            == "ReduceLROnPlateau"
        ):

            self.scheduler.step(
                val_loss
            )

        else:

            self.scheduler.step()

    # ========================================================
    # Save checkpoint
    # ========================================================

    def _save_checkpoint(
        self,
        epoch,
        val_loss,
        val_metrics,
    ):

        checkpoint = {
            "epoch": epoch,

            "model_state_dict": (
                self.model.state_dict()
            ),

            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),

            "scheduler_state_dict": (
                self.scheduler.state_dict()
            ),

            "best_val_loss": (
                self.best_val_loss
            ),

            "best_val_macro_f1": (
                self.best_val_macro_f1
            ),

            "val_loss": val_loss,

            "val_accuracy": (
                val_metrics.accuracy
            ),

            "val_macro_f1": (
                val_metrics.macro_f1
            ),

            "val_macro_precision": (
                val_metrics.macro_precision
            ),

            "val_macro_recall": (
                val_metrics.macro_recall
            ),

            "val_weighted_f1": (
                val_metrics.weighted_f1
            ),

            "val_roc_auc": (
                val_metrics.roc_auc
            ),

            "strategy": self.strategy,

            "class_names": self.class_names,

            "experiment_config": (
                self.experiment_config
            ),
        }

        checkpoint_path = (
            self.output_dir
            / "best_model.pt"
        )

        torch.save(
            checkpoint,
            checkpoint_path,
        )

    # ========================================================
    # Save history
    # ========================================================

    def _save_history(self):

        history_path = (
            self.output_dir
            / "history.pt"
        )

        torch.save(
            self.history,
            history_path,
        )

    # ========================================================
    # Train
    # ========================================================

    def train(self):

        print()
        print("=" * 70)
        print("TRAINING")
        print("=" * 70)

        print(
            f"Strategy: "
            f"{self.strategy}"
        )

        print(
            f"Epochs: "
            f"{self.epochs}"
        )

        print(
            f"Train samples: "
            f"{len(self.train_loader.dataset)}"
        )

        print(
            f"Validation samples: "
            f"{len(self.val_loader.dataset)}"
        )

        print("=" * 70)

        for epoch in range(
            1,
            self.epochs + 1,
        ):

            train_loss, train_metrics = (
                self.train_epoch()
            )

            val_loss, val_metrics = (
                self.validate()
            )

            self._scheduler_step(
                val_loss
            )

            current_lr = (
                self.optimizer
                .param_groups[0]["lr"]
            )

            # ------------------------------------------------
            # History
            # ------------------------------------------------

            epoch_result = {
                "epoch": epoch,

                "train_loss": train_loss,

                "train_accuracy": (
                    train_metrics.accuracy
                ),

                "train_macro_f1": (
                    train_metrics.macro_f1
                ),

                "train_macro_precision": (
                    train_metrics.macro_precision
                ),

                "train_macro_recall": (
                    train_metrics.macro_recall
                ),

                "train_weighted_f1": (
                    train_metrics.weighted_f1
                ),

                "train_roc_auc": (
                    train_metrics.roc_auc
                ),

                "val_loss": val_loss,

                "val_accuracy": (
                    val_metrics.accuracy
                ),

                "val_macro_f1": (
                    val_metrics.macro_f1
                ),

                "val_macro_precision": (
                    val_metrics.macro_precision
                ),

                "val_macro_recall": (
                    val_metrics.macro_recall
                ),

                "val_weighted_f1": (
                    val_metrics.weighted_f1
                ),

                "val_roc_auc": (
                    val_metrics.roc_auc
                ),

                "lr": current_lr,
            }

            self.history.append(
                epoch_result
            )

            self._save_history()

            # ------------------------------------------------
            # Best model
            # ------------------------------------------------

            if (
                val_metrics.macro_f1
                > self.best_val_macro_f1
            ):

                self.best_val_macro_f1 = (
                    val_metrics.macro_f1
                )

                self.best_val_loss = (
                    val_loss
                )

                self._save_checkpoint(
                    epoch=epoch,
                    val_loss=val_loss,
                    val_metrics=val_metrics,
                )

                best_marker = "  <-- BEST"

            else:

                best_marker = ""

            # ------------------------------------------------
            # Epoch output
            # ------------------------------------------------

            print(
                f"Epoch "
                f"{epoch:02d}/{self.epochs} | "
                f"Train Loss: "
                f"{train_loss:.4f} | "
                f"Train F1: "
                f"{train_metrics.macro_f1:.4f} | "
                f"Val Loss: "
                f"{val_loss:.4f} | "
                f"Val F1: "
                f"{val_metrics.macro_f1:.4f} | "
                f"Val AUC: "
                f"{val_metrics.roc_auc:.4f} | "
                f"LR: "
                f"{current_lr:.2e}"
                f"{best_marker}"
            )

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("BEST MODEL")
        print("=" * 70)

        print(
            f"Best validation Macro-F1: "
            f"{self.best_val_macro_f1:.4f}"
        )

        print(
            f"Checkpoint: "
            f"{self.output_dir / 'best_model.pt'}"
        )

        print("=" * 70)

        return self.history