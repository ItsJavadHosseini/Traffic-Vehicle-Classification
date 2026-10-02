import sys
import argparse
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml

from src.data.dataset import VehicleDataset
from src.data.data_module import DataModule
from src.data.resnet_transforms import (
    get_resnet_train_transform,
    get_resnet_eval_transform,
)
from src.models.m_resnet import ResNet18Model
from src.training.resnet_trainer import ResNetTrainer


# ============================================================
# Seed
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Runner
# ============================================================

class ResNetExperimentRunner:

    def __init__(
        self,
        config_path,
        experiment_name,
    ):

        self.config_path = Path(
            config_path
        )

        with open(
            self.config_path,
            "r",
            encoding="utf-8",
        ) as file:

            self.config = yaml.safe_load(
                file
            )

        self.experiment_name = (
            experiment_name
        )

        if experiment_name not in self.config[
            "experiments"
        ]:

            raise ValueError(
                f"Experiment not found: "
                f"{experiment_name}"
            )

        self.experiment_config = (
            self.config[
                "experiments"
            ][experiment_name]
        )

        self.seed = self.config[
            "seed"
        ]

        set_seed(self.seed)

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.output_dir = (
            Path("ARTIFACTS")
            / "resnet"
            / experiment_name
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # Model
    # ========================================================

    def create_model(self):

        model_config = (
            self.experiment_config[
                "model"
            ]
        )

        model = ResNet18Model(
            num_classes=8,
            strategy=model_config[
                "strategy"
            ],
            pretrained=model_config[
                "pretrained"
            ],
            small_input=model_config[
                "small_input"
            ],
        )

        model = model.to(
            self.device
        )

        return model

    # ========================================================
    # Data
    # ========================================================

    def create_dataloaders(self):

        data_config = (
            self.experiment_config[
                "data"
            ]
        )

        split_config = (
            self.config[
                "split"
            ]
        )

        batch_size = data_config[
            "batch_size"
        ]

        image_size = data_config[
            "image_size"
        ]

        data_dir = self.config[
            "dataset"
        ][
            "root_dir"
        ]

        val_size = split_config[
            "val_size"
        ]

        test_size = split_config[
            "test_size"
        ]

        # ----------------------------------------------------
        # Base dataset
        #
        # Used only for creating the split.
        # ----------------------------------------------------

        dataset = VehicleDataset(
            root_dir=data_dir
        )

        data_module = DataModule(
            dataset=dataset,
            val_size=val_size,
            test_size=test_size,
            batch_size=batch_size,
            image_size=(
                image_size,
                image_size,
            ),
            random_state=self.seed,
            num_workers=0,
        )

        data_module.split()

        self.train_indices = (
            data_module.train_indices
        )

        self.val_indices = (
            data_module.val_indices
        )

        # ----------------------------------------------------
        # Transforms
        # ----------------------------------------------------

        train_transform = (
            get_resnet_train_transform(
                image_size=image_size
            )
        )

        eval_transform = (
            get_resnet_eval_transform(
                image_size=image_size
            )
        )

        # ----------------------------------------------------
        # Train dataset
        # ----------------------------------------------------

        train_dataset = VehicleDataset(
            root_dir=data_dir,
            transform=train_transform,
        )

        # ----------------------------------------------------
        # Validation dataset
        # ----------------------------------------------------

        val_dataset = VehicleDataset(
            root_dir=data_dir,
            transform=eval_transform,
        )

        # ----------------------------------------------------
        # Subsets
        # ----------------------------------------------------

        train_subset = torch.utils.data.Subset(
            train_dataset,
            self.train_indices,
        )

        val_subset = torch.utils.data.Subset(
            val_dataset,
            self.val_indices,
        )

        # ----------------------------------------------------
        # DataLoaders
        # ----------------------------------------------------

        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        # ----------------------------------------------------
        # Store information
        # ----------------------------------------------------

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.class_names = (
            dataset.classes
        )

        return (
            train_loader,
            val_loader,
        )

    # ========================================================
    # Checkpoint
    # ========================================================

    def get_checkpoint_path(self):

        checkpoint_config = (
            self.experiment_config[
                "checkpoint"
            ]
        )

        load_from = checkpoint_config.get(
            "load_from"
        )

        if load_from is None:

            return None

        checkpoint_path = (
            Path("ARTIFACTS")
            / "resnet"
            / load_from
            / "best_model.pt"
        )

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                f"Checkpoint for "
                f"{load_from} not found:\n"
                f"{checkpoint_path}"
            )

        return checkpoint_path

    # ========================================================
    # Print Model Information
    # ========================================================

    def print_model_information(
        self,
        model,
    ):

        model_config = (
            self.experiment_config[
                "model"
            ]
        )

        optimizer_config = (
            self.experiment_config[
                "optimizer"
            ]
        )

        print("\n" + "=" * 70)
        print("MODEL INFORMATION")
        print("=" * 70)

        print(
            f"Strategy: "
            f"{model_config['strategy']}"
        )

        print(
            f"Pretrained: "
            f"{model_config['pretrained']}"
        )

        print(
            f"Small input: "
            f"{model_config['small_input']}"
        )

        print(
            f"Total parameters: "
            f"{model.get_total_parameter_count():,}"
        )

        print(
            f"Trainable parameters: "
            f"{model.get_trainable_parameter_count():,}"
        )

        # ----------------------------------------------------
        # Learning rate groups
        # ----------------------------------------------------

        strategy = model_config[
            "strategy"
        ]

        if strategy in {
            "fine_tune_stage_1",
            "fine_tune_stage_2",
        }:

            parameter_groups_config = (
                optimizer_config.get(
                    "parameter_groups",
                    {},
                )
            )

            if strategy == "fine_tune_stage_1":

                parameter_groups = (
                    model.get_parameter_groups(
                        layer4_lr=(
                            parameter_groups_config
                            .get(
                                "layer4",
                                {}
                            )
                            .get(
                                "lr",
                                1e-4,
                            )
                        ),
                        fc_lr=(
                            parameter_groups_config
                            .get(
                                "fc",
                                {}
                            )
                            .get(
                                "lr",
                                1e-3,
                            )
                        ),
                    )
                )

            else:

                parameter_groups = (
                    model.get_parameter_groups(
                        layer3_lr=(
                            parameter_groups_config
                            .get(
                                "layer3",
                                {}
                            )
                            .get(
                                "lr",
                                1e-5,
                            )
                        ),
                        layer4_lr=(
                            parameter_groups_config
                            .get(
                                "layer4",
                                {}
                            )
                            .get(
                                "lr",
                                1e-4,
                            )
                        ),
                        fc_lr=(
                            parameter_groups_config
                            .get(
                                "fc",
                                {}
                            )
                            .get(
                                "lr",
                                1e-3,
                            )
                        ),
                    )
                )

        else:

            parameter_groups = [
                {
                    "params":
                        model.get_trainable_parameters(),
                    "lr":
                        optimizer_config["lr"],
                }
            ]

        print(
            f"LR groups: "
            f"{len(parameter_groups)}"
        )

        for index, group in enumerate(
            parameter_groups,
            start=1,
        ):

            parameter_count = sum(
                parameter.numel()
                for parameter in group[
                    "params"
                ]
            )

            print(
                f"  Group {index}: "
                f"LR={group['lr']:.1e}, "
                f"Params={parameter_count:,}"
            )

    # ========================================================
    # DataLoader Test
    # ========================================================

    def print_dataloader_information(
        self,
    ):

        print("\n" + "=" * 70)
        print("DATA LOADER TEST")
        print("=" * 70)

        train_images, train_labels = next(
            iter(self.train_loader)
        )

        val_images, val_labels = next(
            iter(self.val_loader)
        )

        print(
            f"Train images: "
            f"{train_images.shape}"
        )

        print(
            f"Val images: "
            f"{val_images.shape}"
        )

        print(
            f"Train labels: "
            f"{train_labels.shape}"
        )

        print(
            f"Val labels: "
            f"{val_labels.shape}"
        )

        print(
            f"Train samples: "
            f"{len(self.train_loader.dataset):,}"
        )

        print(
            f"Val samples: "
            f"{len(self.val_loader.dataset):,}"
        )

    # ========================================================
    # Run
    # ========================================================

    def run(self):

        print("\n" + "=" * 70)
        print(
            f"Running experiment: "
            f"{self.experiment_name}"
        )
        print("=" * 70)

        print(
            f"Name: "
            f"{self.experiment_config['name']}"
        )

        print(
            f"Seed: "
            f"{self.seed}"
        )

        print(
            f"Device: "
            f"{self.device}"
        )

        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        model = self.create_model()

        self.print_model_information(
            model
        )

        # ----------------------------------------------------
        # Create data
        # ----------------------------------------------------

        self.create_dataloaders()

        self.print_dataloader_information()

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        checkpoint_path = (
            self.get_checkpoint_path()
        )

        if checkpoint_path is not None:

            print(
                f"\nCheckpoint source: "
                f"{checkpoint_path}"
            )

        else:

            print(
                "\nCheckpoint source: None"
            )

        # ----------------------------------------------------
        # Trainer
        # ----------------------------------------------------

        trainer = ResNetTrainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            experiment_config=(
                self.experiment_config
            ),
            device=self.device,
            class_names=self.class_names,
            output_dir=self.output_dir,
            checkpoint_path=checkpoint_path,
            load_best=(
                self.experiment_config[
                    "checkpoint"
                ].get(
                    "load_best",
                    False,
                )
            ),
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        history = trainer.train()

        return history


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment name, e.g. R1",
    )

    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "resnet_experiments.yaml"
        ),
        help="Path to YAML config",
    )

    args = parser.parse_args()

    runner = ResNetExperimentRunner(
        config_path=args.config,
        experiment_name=args.experiment,
    )

    runner.run()


if __name__ == "__main__":

    main()