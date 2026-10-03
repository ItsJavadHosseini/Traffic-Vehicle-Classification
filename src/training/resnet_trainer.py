from src.evaluation.evaluator import Evaluator
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


class ResNetTrainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        experiment_config,
        device,
        class_names,
        output_dir,
        checkpoint_path=None,
        load_best=False,
    ):
        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.experiment_config = experiment_config

        self.device = device

        self.class_names = class_names

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.epochs = (
            experiment_config[
                "training"
            ]["epochs"]
        )

        self.optimizer_config = (
            experiment_config[
                "optimizer"
            ]
        )

        self.scheduler_config = (
            experiment_config[
                "scheduler"
            ]
        )

        self.strategy = self.model.strategy

        self.criterion = nn.CrossEntropyLoss()
        self.evaluator = Evaluator()

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        self.optimizer = (
            self._create_optimizer()
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        self.scheduler = (
            self._create_scheduler()
        )

        # ----------------------------------------------------
        # Training state
        # ----------------------------------------------------

        self.history = []
        self.best_val_macro_f1 = float("-inf")
        self.best_val_loss = float("inf")

        self.best_epoch = None

        self.total_training_time = 0.0

        # ----------------------------------------------------
        # Load checkpoint if requested
        # ----------------------------------------------------

        if checkpoint_path is not None:

            self.load_checkpoint(
                checkpoint_path=checkpoint_path,
                load_best=load_best,
            )

    # ========================================================
    # Optimizer
    # ========================================================

    def _create_optimizer(self):

        optimizer_name = (
            self.optimizer_config["name"]
        )

        if optimizer_name != "AdamW":

            raise ValueError(
                f"Unsupported optimizer: "
                f"{optimizer_name}"
            )

        weight_decay = (
            self.optimizer_config[
                "weight_decay"
            ]
        )

        # ----------------------------------------------------
        # R3
        #
        # layer3 + layer4 + FC
        # ----------------------------------------------------

        if self.strategy == "fine_tune_stage_2":

            parameter_groups_config = (
                self.optimizer_config.get(
                    "parameter_groups",
                    {},
                )
            )

            layer3_lr = (
                parameter_groups_config
                .get(
                    "layer3",
                    {},
                )
                .get(
                    "lr",
                    1e-5,
                )
            )

            layer4_lr = (
                parameter_groups_config
                .get(
                    "layer4",
                    {},
                )
                .get(
                    "lr",
                    1e-4,
                )
            )

            fc_lr = (
                parameter_groups_config
                .get(
                    "fc",
                    {},
                )
                .get(
                    "lr",
                    1e-3,
                )
            )

            parameter_groups = (
                self.model.get_parameter_groups(
                    layer3_lr=layer3_lr,
                    layer4_lr=layer4_lr,
                    fc_lr=fc_lr,
                )
            )

            return torch.optim.AdamW(
                parameter_groups,
                weight_decay=weight_decay,
            )

        # ----------------------------------------------------
        # R2
        #
        # layer4 + FC
        # ----------------------------------------------------

        if self.strategy == "fine_tune_stage_1":

            parameter_groups_config = (
                self.optimizer_config.get(
                    "parameter_groups",
                    {},
                )
            )

            layer4_lr = (
                parameter_groups_config
                .get(
                    "layer4",
                    {},
                )
                .get(
                    "lr",
                    1e-4,
                )
            )

            fc_lr = (
                parameter_groups_config
                .get(
                    "fc",
                    {},
                )
                .get(
                    "lr",
                    1e-3,
                )
            )

            parameter_groups = (
                self.model.get_parameter_groups(
                    layer4_lr=layer4_lr,
                    fc_lr=fc_lr,
                )
            )

            return torch.optim.AdamW(
                parameter_groups,
                weight_decay=weight_decay,
            )

        # ----------------------------------------------------
        # R1 / R4
        # ----------------------------------------------------

        lr = (
            self.optimizer_config["lr"]
        )

        return torch.optim.AdamW(
            self.model.get_trainable_parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    # ========================================================
    # Scheduler
    # ========================================================

    def _create_scheduler(self):

        scheduler_name = (
            self.scheduler_config["name"]
        )

        if (
            scheduler_name
            == "ReduceLROnPlateau"
        ):

            return (
                torch.optim.lr_scheduler
                .ReduceLROnPlateau(
                    self.optimizer,
                    mode="min",
                )
            )

        if (
            scheduler_name
            == "CosineAnnealingLR"
        ):

            return (
                torch.optim.lr_scheduler
                .CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.epochs,
                )
            )

        raise ValueError(
            f"Unsupported scheduler: "
            f"{scheduler_name}"
        )

    # ========================================================
    # Checkpoint Loading
    # ========================================================

    def load_checkpoint(
        self,
        checkpoint_path,
        load_best=True,
    ):

        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{checkpoint_path}"
            )

        print("\n" + "=" * 70)
        print("LOADING CHECKPOINT")
        print("=" * 70)

        print(
            f"Path: {checkpoint_path}"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        print(
            "Model weights loaded."
        )

        # ----------------------------------------------------
        # Optional optimizer state
        #
        # For R3 we normally DO NOT load the old optimizer
        # because R3 has a different optimizer configuration.
        # ----------------------------------------------------

        if not load_best:

            if (
                "optimizer_state_dict"
                in checkpoint
            ):

                try:

                    self.optimizer.load_state_dict(
                        checkpoint[
                            "optimizer_state_dict"
                        ]
                    )

                    print(
                        "Optimizer state loaded."
                    )

                except ValueError:

                    print(
                        "Warning: optimizer "
                        "state could not be loaded."
                    )

        # ----------------------------------------------------
        # Best validation loss
        # ----------------------------------------------------

        if "best_val_loss" in checkpoint:

            self.best_val_loss = (
                checkpoint[
                    "best_val_loss"
                ]
            )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        checkpoint_epoch = checkpoint.get(
            "epoch",
            None,
        )

        checkpoint_strategy = (
            checkpoint.get(
                "strategy",
                None,
            )
        )

        print(
            f"Checkpoint epoch: "
            f"{checkpoint_epoch}"
        )

        print(
            f"Checkpoint strategy: "
            f"{checkpoint_strategy}"
        )

        print(
            f"Current strategy: "
            f"{self.strategy}"
        )

        print(
            "Checkpoint loading complete."
        )

    # ========================================================
    # Trainable / Frozen mode
    # ========================================================

    def _set_training_mode(self):

        # ----------------------------------------------------
        # R1
        #
        # Frozen backbone.
        # FC in train mode.
        # ----------------------------------------------------

        if self.strategy == "feature_extraction":

            self.model.backbone.eval()

            self.model.backbone.fc.train()
            return

        # ----------------------------------------------------
        # R2
        #
        # Frozen layers + trainable layer4 + FC.
        # ----------------------------------------------------

        if self.strategy == "fine_tune_stage_1":

            self.model.backbone.eval()

            self.model.backbone.layer4.train()

            self.model.backbone.fc.train()

            return

        # ----------------------------------------------------
        # R3
        #
        # Frozen layers + trainable layer3,
        # layer4 and FC.
        # ----------------------------------------------------

        if self.strategy == "fine_tune_stage_2":

            self.model.backbone.eval()

            self.model.backbone.layer3.train()

            self.model.backbone.layer4.train()

            self.model.backbone.fc.train()

            return

        # ----------------------------------------------------
        # R4
        #
        # Everything trainable.
        # ----------------------------------------------------

        if self.strategy == "from_scratch":

            self.model.train()

            return

        raise ValueError(
            f"Unsupported strategy: "
            f"{self.strategy}"
        )

    # ========================================================
    # Learning Rates
    # ========================================================

    def get_learning_rates(self):

        return [
            group["lr"]
            for group in self.optimizer.param_groups
        ]

    # ========================================================
    # Train Epoch
    # ========================================================

    def train_epoch(self):

        self._set_training_mode()

        running_loss = 0.0

        correct = 0

        total = 0

        for images, labels in self.train_loader:

            images = images.to(
                self.device
            )

            labels = labels.to(
                self.device
            )

            # ------------------------------------------------
            # Clear gradients
            # ------------------------------------------------

            self.optimizer.zero_grad()

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            outputs = self.model(
                images
            )

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = self.criterion(
                outputs,
                labels,
            )

            # ------------------------------------------------
            # Backward
            # ------------------------------------------------

            loss.backward()

            # ------------------------------------------------
            # Update weights
            # ------------------------------------------------

            self.optimizer.step()

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            batch_size = (
                labels.size(0)
            )

            running_loss += (
                loss.item()
                * batch_size
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += batch_size

        epoch_loss = (
            running_loss / total
        )

        epoch_accuracy = (
            correct / total
        )

        return (
            epoch_loss,
            epoch_accuracy,
        )

    # ========================================================
    # Validation
    # ========================================================

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = 0.0
        total = 0

        all_targets = []
        all_predictions = []
        all_probabilities = []

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                labels,
            )

            batch_size = labels.size(0)

            running_loss += (
                    loss.item() * batch_size
            )

            total += batch_size

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            predictions = probabilities.argmax(
                dim=1
            )

            all_targets.append(
                labels.cpu().numpy()
            )

            all_predictions.append(
                predictions.cpu().numpy()
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

        val_loss = running_loss / total

        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_predictions)
        y_prob = np.concatenate(all_probabilities)

        metrics = self.evaluator.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
        )

        return val_loss, metrics
    # ========================================================
    # Scheduler Step
    # ========================================================

    def step_scheduler(
        self,
        val_loss,
    ):

        scheduler_name = (
            self.scheduler_config["name"]
        )

        if (
            scheduler_name
            == "ReduceLROnPlateau"
        ):

            self.scheduler.step(
                val_loss
            )

        elif (
            scheduler_name
            == "CosineAnnealingLR"
        ):

            self.scheduler.step()

        else:

            raise ValueError(
                f"Unsupported scheduler: "
                f"{scheduler_name}"
            )

    # ========================================================
    # Save Checkpoint
    # ========================================================

    def save_checkpoint(
                self,
                epoch,
                val_loss,
                val_accuracy,
                val_macro_f1,
        ):

        checkpoint_path = (
            self.output_dir
            / "best_model.pt"
        )

        checkpoint = {

            "epoch":
                epoch,

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "scheduler_state_dict":
                self.scheduler.state_dict(),

            "best_val_loss":
                val_loss,

            "val_accuracy":
                val_accuracy,

            "val_macro_f1":
                val_macro_f1,

            "strategy":
                self.strategy,

            "class_names":
                self.class_names,

            "experiment_config":
                self.experiment_config,
        }
        torch.save(
            checkpoint,
            checkpoint_path,
        )

    # ========================================================
    # Save History
    # ========================================================

    def save_history(self):

        history_path = (
            self.output_dir
            / "history.json"
        )

        with open(
            history_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.history,
                file,
                indent=4,
            )

    # ========================================================
    # Train
    # ========================================================

    def train(self):

        print("\n" + "=" * 70)
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
            f"{len(self.train_loader.dataset):,}"
        )

        print(
            f"Validation samples: "
            f"{len(self.val_loader.dataset):,}"
        )

        training_start = time.time()

        for epoch in range(
            1,
            self.epochs + 1,
        ):

            epoch_start = time.time()

            # ------------------------------------------------
            # Train
            # ------------------------------------------------

            (
                train_loss,
                train_accuracy,
            ) = self.train_epoch()

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            (
                val_loss,
                metrics,
            ) = self.validate()

            val_accuracy = metrics.accuracy
            val_macro_f1 = metrics.macro_f1
            val_macro_precision = metrics.macro_precision
            val_macro_recall = metrics.macro_recall
            val_weighted_f1 = metrics.weighted_f1
            val_roc_auc = metrics.roc_auc

            # ------------------------------------------------
            # Scheduler
            # ------------------------------------------------

            self.step_scheduler(
                val_loss
            )

            # ------------------------------------------------
            # Learning rates
            # ------------------------------------------------

            learning_rates = (
                self.get_learning_rates()
            )

            # ------------------------------------------------
            # Epoch time
            # ------------------------------------------------

            epoch_time = (
                time.time()
                - epoch_start
            )

            # ------------------------------------------------
            # History
            # ------------------------------------------------

            epoch_record = {

                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "train_accuracy":
                    train_accuracy,

                "val_loss":
                    val_loss,

                "val_accuracy":
                    val_accuracy,

                "val_macro_f1":
                    val_macro_f1,

                "val_macro_precision":
                    val_macro_precision,

                "val_macro_recall":
                    val_macro_recall,

                "val_weighted_f1":
                    val_weighted_f1,

                "val_roc_auc":
                    val_roc_auc,

                "learning_rates":
                    learning_rates,

                "epoch_time":
                    epoch_time,
            }
            self.history.append(
                epoch_record
            )

            # ------------------------------------------------
            # Best checkpoint
            # ------------------------------------------------

            is_best = (
                    val_macro_f1
                    > self.best_val_macro_f1
            )

            if is_best:
                self.best_val_macro_f1 = (
                    val_macro_f1
                )

                self.best_epoch = epoch

                self.save_checkpoint(
                    epoch=epoch,
                    val_loss=val_loss,
                    val_accuracy=val_accuracy,
                    val_macro_f1=val_macro_f1,
                )
            # ------------------------------------------------
            # Console output
            # ------------------------------------------------

            lr_text = " | ".join(
                f"LR{i + 1}: {lr:.2e}"
                for i, lr
                in enumerate(
                    learning_rates
                )
            )
            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"| Train Loss: {train_loss:.4f} "
                f"| Train Acc: {train_accuracy:.4f} "
                f"| Val Loss: {val_loss:.4f} "
                f"| Val Acc: {val_accuracy:.4f} "
                f"| Val Macro F1: {val_macro_f1:.4f} "
                f"| {lr_text} "
                f"| Time: {epoch_time:.1f}s"
            )

            # ------------------------------------------------
            # Save history every epoch
            # ------------------------------------------------

            self.save_history()

        # ----------------------------------------------------
        # Total training time
        # ----------------------------------------------------

        self.total_training_time = (
            time.time()
            - training_start
        )

        # ----------------------------------------------------
        # Final summary
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)

        print(
            f"Best epoch: "
            f"{self.best_epoch}"
        )

        print(
            f"Best validation loss: "
            f"{self.best_val_loss:.4f}"
        )

        print(
            f"Total training time: "
            f"{self.total_training_time:.1f}s"
        )

        print(
            f"Best checkpoint: "
            f"{self.output_dir / 'best_model.pt'}"
        )

        return self.history