import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# Resize With Padding
# ============================================================

class ResizeWithPadding:
    """
    Resize an image while preserving its aspect ratio,
    then pad the remaining area to the target size.
    """

    def __init__(self, size=(224, 224), fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, image):

        target_width, target_height = self.size

        width, height = image.size

        scale = min(
            target_width / width,
            target_height / height
        )
        new_width = round(width * scale)
        new_height = round(height * scale)

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.BILINEAR
        )
        pad_left = (
            target_width - new_width
        ) // 2

        pad_top = (
            target_height - new_height
        ) // 2

        pad_right = (
            target_width
            - new_width
            - pad_left
        )

        pad_bottom = (
            target_height
            - new_height
            - pad_top
        )

        image = ImageOps.expand(
            image,
            border=(
                pad_left,
                pad_top,
                pad_right,
                pad_bottom
            ),
            fill=self.fill
        )

        return image


# ============================================================
# Vehicle Subset
# ============================================================

class VehicleSubset(Dataset):
    """
    A subset of the original VehicleDataset.

    The parent dataset keeps the original samples.
    This class only decides which indices belong to
    this particular split and which transform to use.
    """

    def __init__(
        self,
        parent_dataset,
        indices,
        transform=None
    ):
        self.parent_dataset = parent_dataset
        self.indices = np.asarray(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        original_idx = self.indices[idx]

        image_path, label = (
            self.parent_dataset.samples[original_idx]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ============================================================
# Data Module
# ============================================================

class SGDataModule:
    """
    Manage the complete training/validation data pipeline.

    Responsibilities
    ----------------
    1. Create leakage-safe train/validation split
    2. Preserve class distribution
    3. Keep similar images in the same group
    4. Validate the split
    5. Build transforms
    6. Build Dataset objects
    7. Build DataLoaders
    8. Generate reports
    """

    def __init__(
        self,
        dataset,
        val_size=0.15,
        batch_size=32,
        image_size=(224, 224),
        random_state=42,
        num_workers=0,
        train_transform=None,
        eval_transform=None,
        groups=None
    ):

        self.dataset = dataset

        # -----------------------------
        # Configuration
        # -----------------------------

        self.val_size = val_size
        self.batch_size = batch_size
        self.image_size = image_size
        self.random_state = random_state
        self.num_workers = num_workers

        # -----------------------------
        # Optional custom transforms
        # -----------------------------

        self.train_transform = train_transform
        self.eval_transform = eval_transform

        # -----------------------------
        # Group information
        # -----------------------------

        self.groups = groups

        # -----------------------------
        # Split indices
        # -----------------------------

        self.train_indices = None
        self.val_indices = None

        # -----------------------------
        # Dataset objects
        # -----------------------------

        self.train_dataset = None
        self.val_dataset = None

        # -----------------------------
        # DataLoaders
        # -----------------------------

        self.train_loader = None
        self.val_loader = None

        # -----------------------------
        # Reports
        # -----------------------------

        self._split_report = None
        self._validation_report = None

    # ========================================================
    # Groups
    # ========================================================

    def _prepare_groups(self):
        """
        Prepare group IDs for every sample.

        If groups are not provided, every image becomes
        its own group.

        This means:
            image 0 -> group 0
            image 1 -> group 1
            image 2 -> group 2

        If near-duplicate groups are provided, images
        belonging to the same group receive the same ID.
        """

        num_samples = len(self.dataset)

        if self.groups is None:

            self.groups = np.arange(
                num_samples
            )

        else:

            self.groups = np.asarray(
                self.groups
            )

            if len(self.groups) != num_samples:
                raise ValueError(
                    "groups must contain exactly "
                    f"{num_samples} values."
                )

        return self.groups

    # ========================================================
    # Split
    # ========================================================

    def split(self):
        """
        Create a stratified + grouped train/validation split.

        Stratification:
            tries to preserve class proportions.

        Grouping:
            prevents samples belonging to the same group
            from appearing in both train and validation.
        """

        if not 0 < self.val_size < 1:
            raise ValueError(
                "val_size must be between 0 and 1."
            )

        labels = np.array([
            label
            for _, label in self.dataset.samples
        ])

        groups = self._prepare_groups()

        # ----------------------------------------------------
        # Convert desired validation size into number of folds
        # ----------------------------------------------------

        n_splits = round(
            1 / self.val_size
        )

        n_splits = max(
            2,
            n_splits
        )

        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        # ----------------------------------------------------
        # Generate folds
        # ----------------------------------------------------

        splits = list(
            splitter.split(
                X=np.zeros(len(labels)),
                y=labels,
                groups=groups
            )
        )

        # ----------------------------------------------------
        # Select the fold whose validation size is closest
        # to the requested val_size.
        # ----------------------------------------------------

        target_val_size = (
            len(labels) * self.val_size
        )

        best_train_indices = None
        best_val_indices = None
        best_difference = float("inf")

        for train_indices, val_indices in splits:

            difference = abs(
                len(val_indices)
                - target_val_size
            )

            if difference < best_difference:

                best_difference = difference

                best_train_indices = (
                    train_indices
                )

                best_val_indices = (
                    val_indices
                )

        self.train_indices = (
            np.asarray(best_train_indices)
        )

        self.val_indices = (
            np.asarray(best_val_indices)
        )

        self._build_split_report()

        return self._split_report

    # ========================================================
    # Split Report
    # ========================================================

    def _build_split_report(self):

        labels = np.array([
            label
            for _, label in self.dataset.samples
        ])

        records = []

        splits = {
            "train": self.train_indices,
            "validation": self.val_indices
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
                    "percentage": round(
                        float(percentage),
                        2
                    )
                })

        self._split_report = pd.DataFrame(
            records
        )

    @property
    def split_report(self):

        if self._split_report is None:
            self.split()

        return self._split_report

    # ========================================================
    # Validate Split
    # ========================================================

    def validate_split(self):
        """
        Check whether train/validation split is leakage-safe.

        Checks:
        1. Index overlap
        2. Group overlap
        3. Class distribution
        """

        if (
            self.train_indices is None
            or self.val_indices is None
        ):
            raise RuntimeError(
                "Split has not been created yet."
            )

        # ----------------------------------------------------
        # 1. Index overlap
        # ----------------------------------------------------

        train_indices = set(
            self.train_indices
        )

        val_indices = set(
            self.val_indices
        )

        index_overlap = (
            train_indices
            .intersection(val_indices)
        )

        # ----------------------------------------------------
        # 2. Group overlap
        # ----------------------------------------------------

        train_groups = set(
            self.groups[self.train_indices]
        )

        val_groups = set(
            self.groups[self.val_indices]
        )

        group_overlap = (
            train_groups
            .intersection(val_groups)
        )

        # ----------------------------------------------------
        # 3. Class distribution
        # ----------------------------------------------------

        labels = np.array([
            label
            for _, label in self.dataset.samples
        ])

        train_labels = labels[
            self.train_indices
        ]

        val_labels = labels[
            self.val_indices
        ]

        train_distribution = (
            pd.Series(train_labels)
            .value_counts(
                normalize=True
            )
            .sort_index()
        )

        val_distribution = (
            pd.Series(val_labels)
            .value_counts(
                normalize=True
            )
            .sort_index()
        )

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        valid = (
            len(index_overlap) == 0
            and len(group_overlap) == 0
        )

        self._validation_report = {
            "valid": valid,

            "index_overlap": len(
                index_overlap
            ),

            "group_overlap": len(
                group_overlap
            ),

            "train_size": len(
                self.train_indices
            ),

            "validation_size": len(
                self.val_indices
            ),

            "train_groups": len(
                train_groups
            ),

            "validation_groups": len(
                val_groups
            ),

            "train_class_distribution": (
                train_distribution.to_dict()
            ),

            "validation_class_distribution": (
                val_distribution.to_dict()
            )
        }

        if not valid:

            raise ValueError(
                "Data leakage detected.\n"
                f"Index overlap: "
                f"{len(index_overlap)}\n"
                f"Group overlap: "
                f"{len(group_overlap)}"
            )

        return self._validation_report

    # ========================================================
    # Transforms
    # ========================================================

    def build_transforms(self):

        resize_with_padding = (
            ResizeWithPadding(
                size=self.image_size
            )
        )

        # ----------------------------------------------------
        # Default train transform
        # ----------------------------------------------------

        if self.train_transform is None:

            self.train_transform = transforms.Compose([
                resize_with_padding,

                transforms.RandomHorizontalFlip(
                    p=0.5
                ),

                transforms.ToTensor()
            ])

        # ----------------------------------------------------
        # Default validation transform
        # ----------------------------------------------------

        if self.eval_transform is None:

            self.eval_transform = transforms.Compose([
                resize_with_padding,

                transforms.ToTensor()
            ])

        return {
            "train": self.train_transform,
            "validation": self.eval_transform
        }

    # ========================================================
    # Build Datasets
    # ========================================================

    def build_datasets(self):

        if self.train_indices is None:
            self.split()

        if (
            self.train_transform is None
            or self.eval_transform is None
        ):
            self.build_transforms()

        self.train_dataset = VehicleSubset(
            parent_dataset=self.dataset,
            indices=self.train_indices,
            transform=self.train_transform
        )

        self.val_dataset = VehicleSubset(
            parent_dataset=self.dataset,
            indices=self.val_indices,
            transform=self.eval_transform
        )

        return {
            "train": self.train_dataset,
            "validation": self.val_dataset
        }

    # ========================================================
    # Build DataLoaders
    # ========================================================

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

        return {
            "train": self.train_loader,
            "validation": self.val_loader
        }

    # ========================================================
    # Prepare
    # ========================================================

    def prepare(self):

        self.split()

        self.validate_split()

        self.build_transforms()

        self.build_datasets()

        self.build_loaders()

        return self.report()

    # ========================================================
    # Report
    # ========================================================

    def report(self):

        if self.train_loader is None:
            self.prepare()

        return {
            "dataset_size": len(
                self.dataset
            ),

            "num_classes": (
                self.dataset.num_classes
            ),

            "classes": (
                self.dataset.classes
            ),

            "split_sizes": {
                "train": len(
                    self.train_dataset
                ),

                "validation": len(
                    self.val_dataset
                )
            },

            "batch_size": self.batch_size,

            "image_size": self.image_size,

            "num_train_batches": len(
                self.train_loader
            ),

            "num_validation_batches": len(
                self.val_loader
            ),

            "split_distribution": (
                self.split_report
            ),

            "validation": (
                self._validation_report
            )
        }

