import sys
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Project imports
# ============================================================

from src.data.dataset import VehicleDataset
from src.data.data_module import DataModule
from src.models.m_baseline import DeepBaselineCNN
from src.training.experiment_trainer import ExperimentTrainer


# ============================================================
# Paths
# ============================================================

CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments.yaml"
OUTPUT_DIR = PROJECT_ROOT / "ARTIFACTS"


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Config utilities
# ============================================================

def load_config():

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        return yaml.safe_load(f)


def deep_merge(base, override):

    result = deepcopy(base)

    for key, value in override.items():

        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):

            result[key] = deep_merge(
                result[key],
                value,
            )

        else:

            result[key] = deepcopy(value)

    return result


# ============================================================
# Model
# ============================================================

def build_model(
    config,
    num_classes,
):

    model_config = config["model"]

    model_name = model_config["name"]

    if model_name == "deep_baseline":

        return DeepBaselineCNN(
            num_classes=num_classes,
            dropout=model_config["dropout"],
            pooling=model_config.get(
                "pooling",
                "max",
            ),
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


# ============================================================
# Loss
# ============================================================

def build_loss(config):

    loss_name = config["loss"]["name"]

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()

    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()

    raise ValueError(
        f"Unknown loss: {loss_name}"
    )


# ============================================================
# Optimizer
# ============================================================

def build_optimizer(
    config,
    model,
):

    optimizer_config = config["optimizer"]

    optimizer_name = optimizer_config["name"]

    lr = optimizer_config["lr"]

    weight_decay = optimizer_config.get(
        "weight_decay",
        0.0,
    )

    if optimizer_name == "adamw":

        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    if optimizer_name == "adam":

        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    if optimizer_name == "sgd":

        momentum = optimizer_config.get(
            "momentum",
            0.9,
        )

        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    raise ValueError(
        f"Unknown optimizer: {optimizer_name}"
    )


# ============================================================
# Scheduler
# ============================================================

def build_scheduler(
    config,
    optimizer,
):

    scheduler_config = config.get(
        "scheduler",
        {"name": "none"},
    )

    scheduler_name = scheduler_config["name"]

    if scheduler_name == "none":
        return None

    if scheduler_name == "step_lr":

        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config["step_size"],
            gamma=scheduler_config["gamma"],
        )

    if scheduler_name == "reduce_on_plateau":

        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=scheduler_config["patience"],
            factor=scheduler_config["factor"],
        )

    raise ValueError(
        f"Unknown scheduler: {scheduler_name}"
    )


# ============================================================
# Experiment validation
# ============================================================

def validate_experiment_config(config):

    required_sections = [
        "model",
        "loss",
        "optimizer",
        "scheduler",
        "augmentation",
    ]

    for section in required_sections:

        if section not in config:

            raise ValueError(
                f"Missing config section: {section}"
            )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    if config["model"]["name"] not in {
        "deep_baseline",
    }:

        raise ValueError(
            f"Unsupported model: "
            f"{config['model']['name']}"
        )

    if config["model"]["pooling"] not in {
        "max",
        "avg",
    }:

        raise ValueError(
            f"Unsupported pooling: "
            f"{config['model']['pooling']}"
        )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    if config["loss"]["name"] not in {
        "cross_entropy",
        "bce",
    }:

        raise ValueError(
            f"Unsupported loss: "
            f"{config['loss']['name']}"
        )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    if config["optimizer"]["name"] not in {
        "adamw",
        "adam",
        "sgd",
    }:

        raise ValueError(
            f"Unsupported optimizer: "
            f"{config['optimizer']['name']}"
        )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    if config["scheduler"]["name"] not in {
        "none",
        "step_lr",
        "reduce_on_plateau",
    }:

        raise ValueError(
            f"Unsupported scheduler: "
            f"{config['scheduler']['name']}"
        )


# ============================================================
# Print experiment configuration
# ============================================================

def print_experiment_config(config):

    print("\nResolved configuration:")

    print(
        f"  Model:        "
        f"{config['model']['name']}"
    )

    print(
        f"  Pooling:      "
        f"{config['model']['pooling']}"
    )

    print(
        f"  Dropout:      "
        f"{config['model']['dropout']}"
    )

    print(
        f"  Loss:         "
        f"{config['loss']['name']}"
    )

    print(
        f"  Optimizer:    "
        f"{config['optimizer']['name']}"
    )

    print(
        f"  Learning rate:"
        f" {config['optimizer']['lr']}"
    )

    print(
        f"  Weight decay: "
        f"{config['optimizer']['weight_decay']}"
    )

    print(
        f"  Scheduler:    "
        f"{config['scheduler']['name']}"
    )

    print(
        f"  Augmentation: "
        f"{config['augmentation']['enabled']}"
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    config = load_config()

    print(
        f"Config path: {CONFIG_PATH}"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Global seed
    # --------------------------------------------------------

    seed = config["seed"]

    set_seed(seed)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = VehicleDataset(
        root_dir=config["dataset"]["root_dir"]
    )

    print(
        f"Dataset size: {len(dataset)}"
    )

    print(
        f"Classes: {dataset.classes}"
    )

    # --------------------------------------------------------
    # DataModule
    # --------------------------------------------------------

    data_module = DataModule(

        dataset=dataset,

        val_size=config["split"]["val_size"],

        test_size=config["split"].get(
            "test_size",
            0,
        ),

        batch_size=config["data"]["batch_size"],

        image_size=tuple(
            config["data"]["image_size"]
        ),

        random_state=seed,

        num_workers=config["data"].get(
            "num_workers",
            0,
        ),
    )

    # --------------------------------------------------------
    # Diagnostic
    # --------------------------------------------------------

    print(
        f"DataModule module: "
        f"{DataModule.__module__}"
    )

    print(
        f"DataModule has "
        f"build_for_experiment: "
        f"{hasattr(data_module, 'build_for_experiment')}"
    )

    # --------------------------------------------------------
    # Create split ONCE
    # --------------------------------------------------------

    data_module.split()

    print("\nSplit sizes:")

    print(
        f"Train: "
        f"{len(data_module.train_indices)}"
    )

    print(
        f"Validation: "
        f"{len(data_module.val_indices)}"
    )

    print(
        f"Test: "
        f"{len(data_module.test_indices)}"
    )

    # --------------------------------------------------------
    # Experiments
    # --------------------------------------------------------

    experiments = config.get(
        "experiments",
        [],
    )

    if not experiments:

        raise ValueError(
            "No experiments found in experiments.yaml"
        )

    # --------------------------------------------------------
    # Resolve and validate all experiments FIRST
    # --------------------------------------------------------

    defaults = config.get(
        "defaults",
        {},
    )

    resolved_experiments = []

    for experiment in experiments:

        experiment_config = deep_merge(
            defaults,
            experiment,
        )

        validate_experiment_config(
            experiment_config
        )

        resolved_experiments.append(
            experiment_config
        )

    # --------------------------------------------------------
    # Validation summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)

    print(
        f"Validated "
        f"{len(resolved_experiments)} experiments."
    )

    print("=" * 80)

    # ========================================================
    # Run experiments
    # ========================================================

    for experiment_config in resolved_experiments:

        experiment_name = (
            experiment_config["name"]
        )

        # ----------------------------------------------------
        # Experiment header
        # ----------------------------------------------------

        print("\n")
        print("=" * 80)

        print(
            f"STARTING EXPERIMENT: "
            f"{experiment_name}"
        )

        print("=" * 80)

        print_experiment_config(
            experiment_config
        )

        # ----------------------------------------------------
        # Reset seed for every experiment
        # ----------------------------------------------------

        set_seed(seed)

        # ----------------------------------------------------
        # Build data
        # ----------------------------------------------------

        data_module.build_for_experiment(
            augmentation=experiment_config[
                "augmentation"
            ],
        )

        # ----------------------------------------------------
        # Build model
        # ----------------------------------------------------

        model = build_model(
            experiment_config,
            num_classes=dataset.num_classes,
        )

        model = model.to(device)

        # ----------------------------------------------------
        # Build loss
        # ----------------------------------------------------

        criterion = build_loss(
            experiment_config
        )

        # ----------------------------------------------------
        # Build optimizer
        # ----------------------------------------------------

        optimizer = build_optimizer(
            experiment_config,
            model,
        )

        # ----------------------------------------------------
        # Build scheduler
        # ----------------------------------------------------

        scheduler = build_scheduler(
            experiment_config,
            optimizer,
        )

        # ----------------------------------------------------
        # Build trainer
        # ----------------------------------------------------

        trainer = ExperimentTrainer(

            model=model,

            train_loader=data_module.train_loader,

            validation_loader=data_module.val_loader,

            criterion=criterion,

            optimizer=optimizer,

            scheduler=scheduler,

            device=device,

            epochs=config["training"]["epochs"],

            run_name=experiment_name,

            output_dir=OUTPUT_DIR,

            loss_name=experiment_config[
                "loss"
            ]["name"],

            class_names=dataset.classes,

            config=experiment_config,
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        history, metrics = trainer.train()

        # ----------------------------------------------------
        # Experiment summary
        # ----------------------------------------------------

        print("\nExperiment finished:")

        print(
            f"Name: "
            f"{experiment_name}"
        )

        print(
            f"Best epoch: "
            f"{metrics['best_epoch']}"
        )

        print(
            f"Best validation accuracy: "
            f"{metrics['best_val_accuracy']:.4f}"
        )

        print(
            f"Best macro F1: "
            f"{metrics['best_macro_f1']:.4f}"
        )

        print(
            f"Lowest recall class: "
            f"{metrics['lowest_recall_class']}"
        )

        print(
            f"Lowest precision class: "
            f"{metrics['lowest_precision_class']}"
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()