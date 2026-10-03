import numpy as np
import pandas as pd

from PIL import Image, ImageOps

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ResizeWithPadding:

    def __init__(
        self,
        size=(224, 224),
        fill=0,
    ):
        self.size = size
        self.fill = fill

    def __call__(self, image):

        target_width, target_height = self.size

        width, height = image.size

        scale = min(
            target_width / width,
            target_height / height,
        )

        new_width = round(width * scale)
        new_height = round(height * scale)

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.BILINEAR,
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
                pad_bottom,
            ),
            fill=self.fill,
        )

        return image


class DataModule:

    def __init__(
        self,
        dataset,
        val_size=0.2,
        test_size=0.0,
        batch_size=32,
        image_size=(224, 224),
        random_state=42,
        num_workers=0,
    ):

        self.dataset = dataset

        self.val_size = val_size
        self.test_size = test_size

        self.batch_size = batch_size

        self.image_size = image_size

        self.random_state = random_state

        self.num_workers = num_workers

        # ----------------------------------------------------
        # Split indices
        # ----------------------------------------------------

        self.train_indices = None
        self.val_indices = None
        self.test_indices = None

        # ----------------------------------------------------
        # Datasets
        # ----------------------------------------------------

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # ----------------------------------------------------
        # DataLoaders
        # ----------------------------------------------------

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # ----------------------------------------------------
        # Reports
        # ----------------------------------------------------

        self._split_report = None

    # ========================================================
    # Split
    # ========================================================

    def split(self):

        num_samples = len(self.dataset)

        indices = np.arange(num_samples)

        rng = np.random.default_rng(
            self.random_state
        )

        rng.shuffle(indices)

        test_count = int(
            num_samples * self.test_size
        )

        val_count = int(
            num_samples * self.val_size
        )

        self.test_indices = (
            indices[:test_count]
        )

        self.val_indices = (
            indices[
                test_count:
                test_count + val_count
            ]
        )

        self.train_indices = (
            indices[
                test_count + val_count:
            ]
        )

        self._build_split_report()

        return self._split_report

    # ========================================================
    # Split report
    # ========================================================

    def _build_split_report(self):

        labels = np.array(
            [
                label
                for _, label
                in self.dataset.samples
            ]
        )

        records = []

        splits = {
            "train": self.train_indices,
            "validation": self.val_indices,
            "test": self.test_indices,
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

                records.append(
                    {
                        "split": split_name,
                        "class": class_name,
                        "count": int(count),
                        "percentage": float(
                            percentage
                        ),
                    }
                )

        self._split_report = pd.DataFrame(
            records
        )

    @property
    def split_report(self):

        if self._split_report is None:
            self.split()

        return self._split_report

    # ========================================================
    # Normalization
    # ========================================================

    def get_normalize_transform(self):

        return transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        )

    # ========================================================
    # Base transform
    # ========================================================

    def build_base_transform(self):

        resize_with_padding = (
            ResizeWithPadding(
                size=self.image_size
            )
        )

        normalize = (
            self.get_normalize_transform()
        )

        return transforms.Compose(
            [
                resize_with_padding,
                transforms.ToTensor(),
                normalize,
            ]
        )

    # ========================================================
    # Augmented transform
    # ========================================================

    def build_augmented_transform(
        self,
        augmentation_config,
    ):

        normalize = (
            self.get_normalize_transform()
        )

        transform_list = []

        # ----------------------------------------------------
        # Horizontal flip
        # ----------------------------------------------------

        if augmentation_config.get(
            "horizontal_flip",
            False,
        ):

            transform_list.append(
                transforms.RandomHorizontalFlip(
                    p=0.5
                )
            )

        # ----------------------------------------------------
        # Rotation
        # ----------------------------------------------------

        rotation = augmentation_config.get(
            "rotation",
            0,
        )

        if rotation > 0:

            transform_list.append(
                transforms.RandomRotation(
                    degrees=rotation
                )
            )

        # ----------------------------------------------------
        # Color jitter
        # ----------------------------------------------------

        color_jitter = augmentation_config.get(
            "color_jitter",
            False,
        )

        if color_jitter:

            transform_list.append(
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                )
            )

        # ----------------------------------------------------
        # Random crop
        # ----------------------------------------------------

        random_crop = augmentation_config.get(
            "random_crop",
            False,
        )

        if random_crop:

            transform_list.append(
                transforms.RandomResizedCrop(
                    size=self.image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                )
            )

        else:

            transform_list.append(
                ResizeWithPadding(
                    size=self.image_size
                )
            )

        # ----------------------------------------------------
        # Tensor + Normalize
        # ----------------------------------------------------

        transform_list.append(
            transforms.ToTensor()
        )

        transform_list.append(
            normalize
        )

        return transforms.Compose(
            transform_list
        )

    # ========================================================
    # Build transforms
    # ========================================================

    def build_transforms(
        self,
        augmentation=None,
    ):

        if augmentation is None:

            augmentation = {
                "enabled": False
            }

        enabled = augmentation.get(
            "enabled",
            False,
        )

        if enabled:

            self.train_transform = (
                self.build_augmented_transform(
                    augmentation
                )
            )

        else:

            self.train_transform = (
                self.build_base_transform()
            )

        # ----------------------------------------------------
        # Validation is ALWAYS deterministic
        # ----------------------------------------------------

        self.eval_transform = (
            self.build_base_transform()
        )

        return {
            "train": self.train_transform,
            "validation": self.eval_transform,
            "test": self.eval_transform,
        }

    # ========================================================
    # Subset dataset
    # ========================================================

    def _create_subset_dataset(
        self,
        indices,
        transform,
    ):

        parent_dataset = self.dataset

        class VehicleSubset(Dataset):

            def __init__(
                self,
                parent_dataset,
                indices,
                transform,
            ):

                self.parent_dataset = (
                    parent_dataset
                )

                self.indices = indices

                self.transform = transform

            def __len__(self):

                return len(self.indices)

            def __getitem__(
                self,
                idx,
            ):

                original_idx = (
                    self.indices[idx]
                )

                image_path, label = (
                    self.parent_dataset.samples[
                        original_idx
                    ]
                )

                image = Image.open(
                    image_path
                ).convert("RGB")

                if self.transform is not None:

                    image = self.transform(
                        image
                    )

                return image, label

        return VehicleSubset(
            parent_dataset,
            indices,
            transform,
        )

    # ========================================================
    # Build datasets
    # ========================================================

    def build_datasets(self):

        if self.train_indices is None:

            self.split()

        if not hasattr(
            self,
            "train_transform",
        ):

            self.build_transforms()

        self.train_dataset = (
            self._create_subset_dataset(
                self.train_indices,
                self.train_transform,
            )
        )

        self.val_dataset = (
            self._create_subset_dataset(
                self.val_indices,
                self.eval_transform,
            )
        )

        self.test_dataset = (
            self._create_subset_dataset(
                self.test_indices,
                self.eval_transform,
            )
        )

        return {
            "train": self.train_dataset,
            "validation": self.val_dataset,
            "test": self.test_dataset,
        }

    # ========================================================
    # Build loaders
    # ========================================================

    def build_loaders(self):

        if self.train_dataset is None:

            self.build_datasets()

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return {
            "train": self.train_loader,
            "validation": self.val_loader,
            "test": self.test_loader,
        }

    # ========================================================
    # Build data for one experiment
    # ========================================================

    def build_for_experiment(
        self,
        augmentation=None,
    ):

        # ----------------------------------------------------
        # Split MUST already exist.
        #
        # If it doesn't exist, create it once.
        # ----------------------------------------------------

        if self.train_indices is None:

            self.split()

        # ----------------------------------------------------
        # Build transforms according to experiment
        # ----------------------------------------------------

        self.build_transforms(
            augmentation=augmentation
        )

        # ----------------------------------------------------
        # Rebuild datasets
        # using SAME indices
        # ----------------------------------------------------

        self.build_datasets()

        # ----------------------------------------------------
        # Rebuild loaders
        # ----------------------------------------------------

        self.build_loaders()

        return {
            "train": self.train_loader,
            "validation": self.val_loader,
            "test": self.test_loader,
        }

    # ========================================================
    # Prepare
    # ========================================================

    def prepare(
        self,
        augmentation=None,
    ):

        self.split()

        self.build_transforms(
            augmentation=augmentation
        )

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
            "dataset_size":
                len(self.dataset),

            "num_classes":
                self.dataset.num_classes,

            "classes":
                self.dataset.classes,

            "split_sizes": {
                "train":
                    len(self.train_dataset),

                "validation":
                    len(self.val_dataset),

                "test":
                    len(self.test_dataset),
            },

            "batch_size":
                self.batch_size,

            "image_size":
                self.image_size,

            "num_train_batches":
                len(self.train_loader),

            "num_validation_batches":
                len(self.val_loader),

            "num_test_batches":
                len(self.test_loader),

            "split_distribution":
                self.split_report,
        }