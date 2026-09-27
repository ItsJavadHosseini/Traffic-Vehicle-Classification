from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DataModule:
    """
    Prepare VehicleDataset for model training.

    Responsibilities:
    - Train / validation / test split
    - Build transforms
    - Create Dataset objects
    - Create DataLoaders
    - Provide structured reports for visualization
    """

    def __init__(
        self,
        dataset,
        val_size=0.15,
        test_size=0.15,
        batch_size=32,
        image_size=(224, 224),
        random_state=42,
        num_workers=0
    ):
        self.dataset = dataset

        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.image_size = image_size
        self.random_state = random_state
        self.num_workers = num_workers

        self.train_indices = None
        self.val_indices = None
        self.test_indices = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self._split_report = None

    # --------------------------------------------------
    # 1. Split
    # --------------------------------------------------

    def split(self):
        """
        Split dataset into train / validation / test sets.
        """
        num_samples = len(self.dataset)
        indices = np.arange(num_samples)
        rng = np.random.default_rng(self.random_state)
        rng.shuffle(indices)
        test_count = int(num_samples * self.test_size)
        val_count = int(num_samples * self.val_size)
        self.test_indices = indices[:test_count]
        self.val_indices = indices[
            test_count:test_count + val_count
        ]
        self.train_indices = indices[
            test_count + val_count:
        ]
        self._build_split_report()

        return self._split_report

    # --------------------------------------------------
    # 2. Split Report
    # --------------------------------------------------

    def _build_split_report(self):
        labels = np.array([
            label for _, label in self.dataset.samples
        ])

        records = []
        splits = {
            "train": self.train_indices,
            "validation": self.val_indices,
            "test": self.test_indices
        }
        for split_name, indices in splits.items():
            split_labels = labels[indices]
            total = len(indices)
            for class_idx, class_name in enumerate(
                self.dataset.classes
            ):
                count = np.sum(
                    split_labels == class_idx
                )
                percentage = (
                    count / total * 100
                    if total > 0
                    else 0
                )
                records.append({
                    "split": split_name,
                    "class": class_name,
                    "count": int(count),
                    "percentage": float(percentage)
                })

        self._split_report = pd.DataFrame(records)

    @property
    def split_report(self):
        if self._split_report is None:
            self.split()
        return self._split_report

    # --------------------------------------------------
    # 3. Transforms
    # --------------------------------------------------

    def build_transforms(self):
        self.train_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.eval_transform = transforms.Compose([
            transforms.Resize(self.image_size),

            transforms.ToTensor(),
        ])
        return {
            "train": self.train_transform,
            "validation": self.eval_transform,
            "test": self.eval_transform
        }

    # --------------------------------------------------
    # 4. Dataset Subsets
    # --------------------------------------------------

    def _create_subset_dataset(
        self,
        indices,
        transform
    ):
        class VehicleSubset(Dataset):
            def __init__(
                self,
                parent_dataset,
                indices,
                transform
            ):
                self.parent_dataset = parent_dataset
                self.indices = indices
                self.transform = transform

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):

                original_idx = self.indices[idx]

                image_path, label = (
                    self.parent_dataset.samples[original_idx]
                )

                from PIL import Image

                image = Image.open(
                    image_path
                ).convert("RGB")

                if self.transform is not None:
                    image = self.transform(image)

                return image, label

        return VehicleSubset(
            self.dataset,
            indices,
            transform
        )

    # --------------------------------------------------
    # 5. Build Datasets
    # --------------------------------------------------

    def build_datasets(self):

        if self.train_indices is None:
            self.split()

        if not hasattr(self, "train_transform"):
            self.build_transforms()

        self.train_dataset = self._create_subset_dataset(
            self.train_indices,
            self.train_transform
        )

        self.val_dataset = self._create_subset_dataset(
            self.val_indices,
            self.eval_transform
        )

        self.test_dataset = self._create_subset_dataset(
            self.test_indices,
            self.eval_transform
        )
        return {
            "train": self.train_dataset,
            "validation": self.val_dataset,
            "test": self.test_dataset
        }

    # --------------------------------------------------
    # 6. DataLoaders
    # --------------------------------------------------

    def build_loaders(self):

        if self.train_dataset is None:
            self.build_datasets()

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        return {
            "train": self.train_loader,
            "validation": self.val_loader,
            "test": self.test_loader
        }

    # --------------------------------------------------
    # 7. Pipeline
    # --------------------------------------------------

    def prepare(self):

        self.split()

        self.build_transforms()

        self.build_datasets()

        self.build_loaders()

        return self.report()

    # --------------------------------------------------
    # 8. Report
    # --------------------------------------------------

    def report(self):

        if self.train_loader is None:
            self.prepare()

        report = {
            "dataset_size": len(self.dataset),

            "num_classes": self.dataset.num_classes,

            "classes": self.dataset.classes,

            "split_sizes": {
                "train": len(self.train_dataset),
                "validation": len(self.val_dataset),
                "test": len(self.test_dataset)
            },

            "batch_size": self.batch_size,

            "image_size": self.image_size,

            "num_train_batches": len(self.train_loader),

            "num_validation_batches": len(self.val_loader),

            "num_test_batches": len(self.test_loader),

            "split_distribution": self.split_report
        }

        return report