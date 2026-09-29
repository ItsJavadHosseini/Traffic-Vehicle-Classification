import sys
from pathlib import Path
import random
import copy
import itertools

import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.dataset import VehicleDataset
from src.data.data_module import DataModule
from src.models.m_baseline import DeepBaselineCNN
from src.training.experiment_trainer import ExperimentTrainer

from runs.experiments import EXPERIMENTS


# --------------------------------------------------
# Seed
# --------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# Load config
# --------------------------------------------------

CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print("CONFIG PATH:", CONFIG_PATH)
print("CONFIG:", config)
# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")

# --------------------------------------------------
# Seed
# --------------------------------------------------

set_seed(config["seed"])

# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = VehicleDataset(
    root_dir=config["dataset"]["root_dir"]
)

print(f"Dataset size: {len(dataset)}")
print(f"Classes: {dataset.classes}")

# --------------------------------------------------
# DataModule
# --------------------------------------------------

data_module = DataModule(
    dataset=dataset,
    val_size=config["split"]["val_size"],
    test_size=config["split"]["test_size"],
    batch_size=config["data"]["batch_size"],
    image_size=tuple(
        config["data"]["image_size"]
    ),
    random_state=config["seed"],
    num_workers=config["data"]["num_workers"],
)

data_module.prepare()


print("\nSplit sizes:")
print(
    f"Train: {len(data_module.train_dataset)}"
)
print(
    f"Validation: {len(data_module.val_dataset)}"
)
print(
    f"Test: {len(data_module.test_dataset)}"
)

# --------------------------------------------------
# Loss
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

# --------------------------------------------------
# Run experiments
# --------------------------------------------------

for experiment in EXPERIMENTS:

    experiment_name = experiment["name"]

    print("\n" + "=" * 70)
    print(
        f"Starting experiment: "
        f"{experiment_name}"
    )
    print("=" * 70)

    set_seed(config["seed"])

    model = DeepBaselineCNN(
        num_classes=dataset.num_classes,
        dropout=experiment["dropout"],
    )

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=experiment["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
    )

    trainer = ExperimentTrainer(
        model=model,
        train_loader=data_module.train_loader,
        validation_loader=data_module.val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=config["training"]["epochs"],
        run_name=experiment_name,
        output_dir=PROJECT_ROOT / "ARTIFACTS",
        config=experiment,
    )

    history, metrics = trainer.train()

    print("\nExperiment finished:")
    print(
        f"Name: {experiment_name}"
    )
    print(
        f"Best epoch: "
        f"{metrics['best_epoch']}"
    )
    print(
        f"Best validation accuracy: "
        f"{metrics['best_val_accuracy']:.4f}"
    )