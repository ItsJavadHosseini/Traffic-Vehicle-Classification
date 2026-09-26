from pathlib import Path
from collections import Counter
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image,ImageStat,ImageFilter
from torch.utils.data import Dataset, DataLoader
import imagehash

class VehicleDataset(Dataset):
    SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    def __init__(self,root_dir,transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        if not self.root_dir.exists():
            raise FileNotFoundError(f"{self.root_dir} does not exist")

        self.classes = self._find_classes()
        self.class_to_idx = {
            class_name : idx
            for idx, class_name in self.classes
        }
        self.samples = self._load_samles()
        self._metadata = None
        self._corrupted = None
        self._exact_duplicates = None
        self._near_duplicates = None

        # 1. load dataset
        def _find_classes(self):
            """
            Find Classes name from subdirectories.
            :param self:
            :return:
            """
            classes = sorted(folder.name for folder in Path(self.root_dir).iterdir() if\
                             folder.is_dir())
            if not classes:
                raise ValueError(f"No classes found in {self.root_dir}")
            return classes
        def _load_samles(self):
            """
            Create a list of: (image_path, class_index)
            :param self:
            :return:
            """
            samles = []
            for class_name in self.classes:
                class_dir = self.root_dir / class_name
                for path in class_dir.rglob("*"):
                    if (
                        path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
                    ):
                        label = self.class_to_idx[class_name]
                        samles.append((path, label))

            if not samles:
                raise ValueError(f"No classes found in {self.root_dir}")
            return samles
