from pathlib import Path
import argparse
import random
import sys

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.dataset import VehicleDataset
from src.data.data_module import DataModule
from src.models.m_vit import ViTModel
from src.training.vit_trainer import ViTTrainer


class ViTRunner:

    def __init__(self, experiment_name):

        self.experiment_name = experiment_name

        self.config_path = (
            PROJECT_ROOT
            / "configs"
            / "vit_experiments.yaml"
        )

        with open(
            self.config_path,
            "r",
            encoding="utf-8",
        ) as f:
            self.config = yaml.safe_load(f)

        if experiment_name not in self.config["experiments"]:

            available = list(
                self.config["experiments"].keys()
            )

            raise ValueError(
                f"Unknown experiment: {experiment_name}. "
                f"Available experiments: {available}"
            )

        self.experiment_config = (
            self.config["experiments"][
                experiment_name
            ]
        )

        self.seed = self.config.get(
            "seed",
            42,
        )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.output_dir = (
            PROJECT_ROOT
            / "ARTIFACTS"
            / "vit"
            / experiment_name
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # Seed
    # ========================================================

    def set_seed(self):

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():

            torch.cuda.manual_seed_all(
                self.seed
            )

    # ========================================================
    # Dataset
    # ========================================================

    def build_dataset(self):

        dataset_config = self.config["dataset"]

        dataset = VehicleDataset(
            root_dir=(
                PROJECT_ROOT
                / dataset_config["root_dir"]
            )
        )

        return dataset

    # ========================================================
    # Data Module
    # ========================================================

    def build_data_module(self, dataset):

        split_config = self.config["split"]
        data_config = self.experiment_config["data"]

        image_size = data_config.get(
            "image_size",
            224,
        )

        data_module = DataModule(
            dataset=dataset,
            val_size=split_config["val_size"],
            test_size=split_config["test_size"],
            batch_size=data_config.get(
                "batch_size",
                16,
            ),
            image_size=(
                image_size,
                image_size,
            ),
            random_state=self.seed,
            num_workers=0,
        )

        return data_module

    # ========================================================
    # Model
    # ========================================================

    def build_model(self):

        model_config = self.experiment_config["model"]

        model = ViTModel(
            num_classes=8,
            strategy=model_config["strategy"],
            pretrained=model_config["pretrained"],
        )

        return model

    # ========================================================
    # Experiment Information
    # ========================================================

    def print_experiment_info(
        self,
        model,
        dataset,
    ):

        model_config = self.experiment_config["model"]
        data_config = self.experiment_config["data"]
        training_config = self.experiment_config["training"]

        print()
        print("=" * 60)
        print("ViT EXPERIMENT")
        print("=" * 60)

        print(
            f"Experiment       : "
            f"{self.experiment_name}"
        )

        print(
            f"Device            : "
            f"{self.device}"
        )

        print(
            "Model             : "
            "ViT-B/16"
        )

        print(
            f"Strategy          : "
            f"{model_config['strategy']}"
        )

        print(
            f"Pretrained        : "
            f"{model_config['pretrained']}"
        )

        print(
            f"Image size        : "
            f"{data_config.get('image_size', 224)}"
        )

        print(
            f"Batch size        : "
            f"{data_config.get('batch_size', 16)}"
        )

        print(
            f"Epochs            : "
            f"{training_config['epochs']}"
        )

        print(
            f"Total parameters  : "
            f"{model.get_total_parameter_count():,}"
        )

        print(
            f"Trainable params  : "
            f"{model.get_trainable_parameter_count():,}"
        )

        print(
            f"Classes           : "
            f"{len(dataset.classes)}"
        )

        print(
            f"Class names       : "
            f"{dataset.classes}"
        )

        print(
            f"Output directory  : "
            f"{self.output_dir}"
        )

        print("=" * 60)

    # ========================================================
    # Run
    # ========================================================

    def run(self):

        self.set_seed()

        print("Creating model...")
        model = self.build_model()

        print("Creating dataloaders...")
        dataset = self.build_dataset()

        data_module = self.build_data_module(
            dataset
        )

        loaders = data_module.build_for_experiment(
            augmentation=None
        )

        train_loader = loaders["train"]
        val_loader = loaders["validation"]
        test_loader = loaders["test"]

        model = model.to(self.device)

        self.print_experiment_info(
            model,
            dataset,
        )

        # ----------------------------------------------------
        # Data information
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("DATA")
        print("=" * 70)

        print(
            f"Train samples      : "
            f"{len(train_loader.dataset)}"
        )

        print(
            f"Validation samples : "
            f"{len(val_loader.dataset)}"
        )

        print(
            f"Test samples       : "
            f"{len(test_loader.dataset)}"
        )

        print("=" * 70)

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("TRAINING")
        print("=" * 70)

        trainer = ViTTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            experiment_config=self.experiment_config,
            device=self.device,
            class_names=dataset.classes,
            output_dir=self.output_dir,
        )

        history = trainer.train()

        # ----------------------------------------------------
        # Training results
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("TRAINING FINISHED")
        print("=" * 70)

        if history:

            best_epoch = max(
                history,
                key=lambda x: x["val_macro_f1"],
            )

            print(
                f"Best epoch       : "
                f"{best_epoch['epoch']}"
            )

            print(
                f"Best val accuracy: "
                f"{best_epoch['val_accuracy']:.4f}"
            )

            print(
                f"Best val macro F1: "
                f"{best_epoch['val_macro_f1']:.4f}"
            )

        print(
            f"Artifacts saved  : "
            f"{self.output_dir}"
        )

        print("=" * 70)


# ============================================================
# Argument Parser
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Run ViT experiments."
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help=(
            "Experiment name defined in "
            "vit_experiments.yaml"
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    runner = ViTRunner(
        experiment_name=args.experiment
    )

    runner.run()


if __name__ == "__main__":
    main()