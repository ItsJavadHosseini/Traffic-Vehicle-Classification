from pathlib import Path
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


class ViTEnergyEvaluator:

    def __init__(self):
        self.experiment_name = "V1"

        self.config_path = (
            PROJECT_ROOT
            / "configs"
            / "vit_experiments.yaml"
        )

        self.checkpoint_path = (
            PROJECT_ROOT
            / "ARTIFACTS"
            / "vit"
            / "V1"
            / "best_model.pt"
        )

        with open(
            self.config_path,
            "r",
            encoding="utf-8",
        ) as f:
            self.config = yaml.safe_load(f)

        self.experiment_config = (
            self.config["experiments"][
                self.experiment_name
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

        return DataModule(
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

    # ========================================================
    # Model
    # ========================================================

    def build_model(self, dataset):

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )

        model = ViTModel(
            num_classes=len(dataset.classes),
            strategy=checkpoint["strategy"],
            pretrained=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model = model.to(self.device)
        model.eval()

        return model, checkpoint

    # ========================================================
    # Energy
    # ========================================================

    @staticmethod
    def calculate_energy(logits):

        return -torch.logsumexp(
            logits,
            dim=1,
        )

    # ========================================================
    # Collect Energy
    # ========================================================

    @torch.no_grad()
    def collect_energy(
        self,
        model,
        dataloader,
    ):

        all_energy = []
        all_targets = []

        for images, targets in dataloader:

            images = images.to(self.device)

            logits = model(images)

            energy = self.calculate_energy(
                logits
            )

            all_energy.append(
                energy.cpu().numpy()
            )

            all_targets.append(
                targets.numpy()
            )

        energies = np.concatenate(
            all_energy
        )

        targets = np.concatenate(
            all_targets
        )

        return energies, targets

    # ========================================================
    # Statistics
    # ========================================================

    def print_statistics(
        self,
        energies,
        targets,
        class_names,
    ):

        print()
        print("=" * 70)
        print("ViT ENERGY SCORE ANALYSIS")
        print("=" * 70)

        print(
            f"Samples       : {len(energies)}"
        )

        print(
            f"Min Energy    : {energies.min():.6f}"
        )

        print(
            f"Max Energy    : {energies.max():.6f}"
        )

        print(
            f"Mean Energy   : {energies.mean():.6f}"
        )

        print(
            f"Median Energy : {np.median(energies):.6f}"
        )

        print()
        print("Percentiles:")
        print(
            f"P01 : {np.percentile(energies, 1):.6f}"
        )
        print(
            f"P05 : {np.percentile(energies, 5):.6f}"
        )
        print(
            f"P50 : {np.percentile(energies, 50):.6f}"
        )
        print(
            f"P90 : {np.percentile(energies, 90):.6f}"
        )
        print(
            f"P95 : {np.percentile(energies, 95):.6f}"
        )
        print(
            f"P99 : {np.percentile(energies, 99):.6f}"
        )

        # ----------------------------------------------------
        # Preliminary threshold
        # ----------------------------------------------------

        threshold = np.percentile(
            energies,
            95,
        )

        print()
        print("=" * 70)
        print("PRELIMINARY THRESHOLD")
        print("=" * 70)

        print(
            f"Threshold (P95): {threshold:.6f}"
        )

        print(
            "Decision rule:"
        )

        print(
            f"Energy <= {threshold:.6f}  -> KNOWN"
        )

        print(
            f"Energy >  {threshold:.6f}  -> UNKNOWN"
        )

        # ----------------------------------------------------
        # Known rejection rate
        # ----------------------------------------------------

        known_rejected = (
            energies > threshold
        )

        rejection_rate = (
            known_rejected.mean()
        )

        print()
        print(
            "Known images classified as "
            f"UNKNOWN: {rejection_rate:.2%}"
        )

        print("=" * 70)

        return threshold

    # ========================================================
    # Per-class Statistics
    # ========================================================

    def print_class_statistics(
        self,
        energies,
        targets,
        class_names,
    ):

        print()
        print("=" * 70)
        print("ENERGY BY CLASS")
        print("=" * 70)

        for class_index, class_name in enumerate(
            class_names
        ):

            class_energies = energies[
                targets == class_index
            ]

            if len(class_energies) == 0:
                continue

            print()
            print(
                f"{class_name}:"
            )

            print(
                f"  Count  : {len(class_energies)}"
            )

            print(
                f"  Mean   : "
                f"{class_energies.mean():.6f}"
            )

            print(
                f"  Median : "
                f"{np.median(class_energies):.6f}"
            )

            print(
                f"  P95    : "
                f"{np.percentile(class_energies, 95):.6f}"
            )

            print(
                f"  Max    : "
                f"{class_energies.max():.6f}"
            )

    # ========================================================
    # Run
    # ========================================================

    def run(self):

        self.set_seed()

        print()
        print("=" * 70)
        print("BUILDING DATASET")
        print("=" * 70)

        dataset = self.build_dataset()

        print(
            f"Classes: {dataset.classes}"
        )

        print(
            f"Total samples: {len(dataset)}"
        )

        print()
        print("=" * 70)
        print("BUILDING VALIDATION SPLIT")
        print("=" * 70)

        data_module = self.build_data_module(
            dataset
        )

        loaders = (
            data_module.build_for_experiment(
                augmentation=None
            )
        )

        val_loader = loaders["validation"]

        print(
            f"Validation samples: "
            f"{len(val_loader.dataset)}"
        )

        print()
        print("=" * 70)
        print("LOADING BEST ViT CHECKPOINT")
        print("=" * 70)

        print(
            f"Checkpoint:\n"
            f"{self.checkpoint_path}"
        )

        model, checkpoint = self.build_model(
            dataset
        )

        print(
            f"Epoch: "
            f"{checkpoint.get('epoch')}"
        )

        print(
            f"Strategy: "
            f"{checkpoint.get('strategy')}"
        )

        print(
            f"Device: {self.device}"
        )

        print()
        print("=" * 70)
        print("CALCULATING ENERGY")
        print("=" * 70)

        energies, targets = self.collect_energy(
            model=model,
            dataloader=val_loader,
        )

        threshold = self.print_statistics(
            energies=energies,
            targets=targets,
            class_names=dataset.classes,
        )

        self.print_class_statistics(
            energies=energies,
            targets=targets,
            class_names=dataset.classes,
        )

        print()
        print("=" * 70)
        print("DONE")
        print("=" * 70)

        print(
            f"Preliminary threshold: "
            f"{threshold:.6f}"
        )


def main():

    evaluator = ViTEnergyEvaluator()
    evaluator.run()


if __name__ == "__main__":
    main()