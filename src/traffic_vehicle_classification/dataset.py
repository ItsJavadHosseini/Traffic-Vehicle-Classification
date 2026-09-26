
from pathlib import Path
from collections import Counter
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import imagehash

class VehicleDataset(Dataset):
    SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    def __init__(self,root_dir,transform=None):
        #self.near_duplicates = None
        #self.exact_duplicates = None
        self.root_dir = Path(root_dir)
        self.transform = transform

        if not self.root_dir.exists():
            raise FileNotFoundError(f"{self.root_dir} does not exist")

        self.classes = self._find_classes()
        self.class_to_idx = {
            class_name : idx
            for idx, class_name in enumerate(self.classes)
        }
        self.samples = self._load_samples()
        self._metadata = None
        self._corrupted = None
        self._exact_duplicates = None
        self._near_duplicates = None

# 1. load dataset

    def _find_classes(self):
        """
        Find Classes name from subdirectories.
        :param self:
        :return: name of classes
        """
        classes = sorted(folder.name for folder in Path(self.root_dir).iterdir() if\
                            folder.is_dir())
        if not classes:
            raise ValueError(f"No classes found in {self.root_dir}")
        return classes
    def _load_samples(self):
        """
        Create a list of: (image_path, class_index) |
        :param self:
        :return:image path ,class index
        """
        samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for path in class_dir.rglob("*"):
                if (
                    path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
                ):
                    label = self.class_to_idx[class_name]
                    samples.append((path, label))

        if not samples:
            raise ValueError(f"No classes found in {self.root_dir}")
        return samples
# 2. Basic Information
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        image_path,label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image,label

    @property
    def num_samples(self):
        return len(self.samples)

    @property
    def num_classes(self):
        return len(self.classes)

    def summary(self):
        """
        print a high-level summary of the dataset
        :param self:
        :return:
        """
        print("Dataset Summary:")
        print(f"Root directory: {self.root_dir}")
        print(f"Number of classes: {self.num_classes}")

        print("Classes")
        for class_name in self.classes:
            count = sum(
                1 for path, label in self.samples if self. classes[label]== class_name
            )
            print(f"{class_name} : {count}")

# 3. 4. MetaDate

    def build_metadata(self, force=False):
        if self._metadata is not None and not force:
            return self._metadata
        records = []
        for image_path, label in self.samples:
            record = {
                "path" : str(image_path),
                "filename" : image_path.name,
                "label" : self.classes[label],
                "format" : image_path.suffix.lower(),
                "filesize" : image_path.stat().st_size
            }
            try:
                with Image.open(image_path) as img:
                    record["width"] = img.width
                    record["height"] = img.height
                    record["channels"] = len(img.getbands())
                    record["aspect_ratio"] = img.width / img.height

            except Exception as e:
                record["width"] = None
                record["height"] = None
                record["channels"] = None
                record["aspect_ratio"] = None
            records.append(record)
        self._metadata = pd.DataFrame(records)
        return self._metadata
    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = self.build_metadata()
        return self._metadata

# 5. Image Properties

    def inspect_images(self):
        df = self.build_metadata()
        print("Image Dimension")
        print(
            df[
                ["width", "height"]
            ].describe()
        )
        print(f"Channels: {df['channels'].value_counts()}")
        print(f"Format: {df['format'].value_counts()}")
        print(f"Filesize: {df['filesize'].value_counts()}")
        print(f"Aspect Ratio: {df['aspect_ratio'].value_counts()}")
        return df

# 6. Corrupted Images

    def check_corrupted(self):
        corrupted = []
        for image_path, label in self.samples:
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception as error:
                corrupted.append(
                    {"path" : image_path, "label" : label, "error" : error}

                )
        self._corrupted = pd.DataFrame(corrupted)
        print(f"Corrupted: {len(self._corrupted)}")
        return self._corrupted

# 7. Visualize Sample

    def visualize_samples(self,sample_per_class=5, fig_size=(15 , 10)):
        fig, axes = plt.subplots(self.num_classes,sample_per_class, figsize=fig_size)
        if self.num_classes == 1:
            axes = np.expand_dims(axes, axis=0)
        if sample_per_class == 1:
            axes = np.expand_dims(axes, axis=1)
        for class_idx, class_name in enumerate(self.classes):
            class_samples = [path for path, label in self.samples if label == class_idx]
            selected = class_samples[:sample_per_class]
            for col, image_path in enumerate(selected):
                ax = axes[class_idx, col]

                try:
                    image = Image.open(image_path).convert("RGB")
                    ax.imshow(image)
                    ax.axis('off')
                    if col == 0:
                        ax.set_title(class_name)
                except Exception:
                    ax.text(0.5, 0.5, "error", ha="center", va="center")
        plt.tight_layout()
        plt.show()

# 8. Class Distribution

    def class_distribution(self):
        count = Counter(self.classes[label] for _, label in self.samples)
        df = pd.DataFrame({
            "class" : list(count.keys()),
            "count" : list(count.values())
        })

        df = df.sort_values("count", ascending=False).reset_index(drop=True)
        print(df)
        return df

# 9. Image Quality Analyses

    def analyze_image_properties(self):
        record = []
        for image_path, label in self.samples:
            try:
                image = Image.open(image_path).convert("L")
                array = np.asarray(image)
                brightness = float(array.mean())

                #variance of laplacian as a simple
                #sharpnes / blur indicator
                laplacian_variance = float(np.var(np.gradient(array.astype(float))))
                record.append({
                    "path" : str(image_path),
                    "label" : self.classes[label],
                    "brightness" : brightness,
                    "laplacian_variance" : laplacian_variance,
                })
            except Exception:
                record.append({
                    "path" : str(image_path),
                    "label" : self.classes[label],
                    "brightness" : None,
                    "laplacian_variance" : None,
                })
        properties = pd.DataFrame(record)
        if self._metadata is None:
            self._metadata = self.build_metadata()
        self._metadata = self._metadata.drop(
            columns=["brightness", "laplacian_variance"],
            errors="ignore"
        )
        self._metadata = self.build_metadata().merge(
                properties[
                ["path", "brightness", "laplacian_variance"]
            ],
            on="path",
            how="left"
        )
        return self._metadata

# 10. Exact Duplicates
    def _md5(self, path):
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def find_exact_duplicates(self):
        records = []
        for image_path, label in  self.samples:
            try:
                file_hash = self._md5(image_path)
                records.append({
                    "path" : str(image_path),
                    "label" : self.classes[label],
                    "md5" : file_hash,
                })
            except Exception:
                continue
        df = pd.DataFrame(records)
        duplicate_mask = df.duplicated(subset="md5", keep=False)
        duplicates = (
            df[duplicate_mask].sort_values("md5").reset_index(drop=True)
        )
        self.exact_duplicates = duplicates
        print(f"Exact Duplicate files: {len(duplicates)}")
        return duplicates

# 11. Near Duplicates

    def find_near_duplicates(self, max_distance=5):
        records = []
        for image_path, label in self.samples:
            try:
                with Image.open(image_path) as image:
                    image_hash = imagehash.phash(image)
                records.append({
                    "path" : str(image_path),
                    "label" : self.classes[label],
                    "hash" : image_hash,
                })
            except Exception:
                continue

        duplicates = []
        for i in range(len(records)):
            for j in range(i+1, len(records)):
                distance = (
                    records[i]["hash"] - records[j]["hash"]
                )
                if distance < max_distance:
                    duplicates.append({
                        "path_1": records[i]["path"],
                        "label_1" : records[i]["label"],
                        "path_2": records[j]["path"],
                        "label_2" : records[j]["label"],
                        "distance" : distance,
                    })
        self.near_duplicates = pd.DataFrame(duplicates)
        print(f"Near Duplicate files: {len(self.near_duplicates)}")
        return self.near_duplicates

# 12 . Suspicious Samples

    def suspicious_samples(self, brightness_threshold=None, blur_threshold=None):
        df = self.analyze_image_properties()
        suspicious = pd.Series(
            False,
            index=df.index,
        )
        if brightness_threshold is not None:
            suspicious |= (
                df["brightness"] < brightness_threshold
            )
        if blur_threshold is not None:
            suspicious |= (df["laplacian_variance"] < blur_threshold)
        result = df[suspicious].copy()
        print(f"Suspicious samples: {len(result)}")
        return result

# 13 . Report

    def audit_report(self):
        meta_data = self.build_metadata()
        report = {
            "root dir" : str(self.root_dir),
            "num classes" : self.num_classes,
            "num images" : self.num_samples,
            "classes" : self.classes,
            "class distribution" : self.class_distribution().to_dict("records"),
            "formats": (meta_data["format"].value_counts().to_dict()),
            "channels" : (meta_data["channels"].value_counts().to_dict()),
            "corrupted images" : len(self.check_corrupted())
        }
        return report

# 14 . DataLoader
    def get_loader(
            self,
            batch_size=32,
            shuffle=True,
            num_workers=0,
    ):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

















