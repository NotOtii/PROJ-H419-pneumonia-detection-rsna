from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PneumoniaDataset(Dataset):
    """
    Custom PyTorch Dataset for binary pneumonia classification.

    This dataset connects:
        - The classification_labels.csv file
        - The train/val/test split files
        - The PNG images stored on disk

    Expected CSV format (classification_labels.csv):
        image_path,label
        data/png/train/xxx.png,1
        data/png/train/yyy.png,0

    Expected split file format (train.txt / val.txt / test.txt):
        data/png/train/xxx.png
        data/png/train/yyy.png
        ...

    Parameters
    ----------
    labels_csv : str or Path
        Path to classification_labels.csv.

    split_file : str or Path
        Path to split text file (train.txt / val.txt / test.txt).

    base_dir : str or Path, optional
        Base directory to resolve image paths (default: ".").

    transform : torchvision.transforms, optional
        Transform pipeline applied to images.
        Train/val/test transformations must be defined externally.
    """

    def __init__(
        self,
        labels_csv: str | Path,
        split_file: str | Path,
        base_dir: str | Path = ".",
        transform=None,
    ):
        self.labels_csv = Path(labels_csv)
        self.split_file = Path(split_file)
        self.base_dir = Path(base_dir)
        self.transform = transform

        # -----------------------------
        # Safety checks
        # -----------------------------
        if not self.labels_csv.exists():
            raise FileNotFoundError(f"labels_csv not found: {self.labels_csv}")

        if not self.split_file.exists():
            raise FileNotFoundError(f"split_file not found: {self.split_file}")

        # -----------------------------
        # Load label table
        # -----------------------------
        df = pd.read_csv(self.labels_csv)

        if "image_path" not in df.columns or "label" not in df.columns:
            raise ValueError("labels_csv must contain columns: image_path,label")

        df["label"] = df["label"].astype(int)

        # -----------------------------
        # Load split list
        # -----------------------------
        split_paths = [
            line.strip()
            for line in self.split_file.read_text(encoding="utf-8").splitlines()
        ]
        split_set = set(Path(p).as_posix() for p in split_paths)

        # Normalize paths for comparison
        df["image_path_norm"] = df["image_path"].apply(lambda p: Path(p).as_posix())

        # Keep only images belonging to the chosen split
        df = df[df["image_path_norm"].isin(split_set)].reset_index(drop=True)

        # -----------------------------
        # Verify that images exist
        # -----------------------------
        def exists(p: str) -> bool:
            return (self.base_dir / Path(p)).exists()

        missing = df[~df["image_path"].apply(exists)]

        if len(missing) > 0:
            print(
                f"[WARNING] {len(missing)} images listed in split_file "
                f"were not found on disk and will be skipped."
            )
            df = df[df["image_path"].apply(exists)].reset_index(drop=True)

        self.df = df

    def __len__(self) -> int:
        """
        Returns the number of samples in this split.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[object, int]:
        """
        Returns:
            image (transformed tensor)
            label (int: 0 or 1)
        """
        row = self.df.iloc[idx]
        rel_path = Path(row["image_path"])
        img_path = self.base_dir / rel_path

        # -----------------------------
        # Load image as grayscale
        # -----------------------------
        img = Image.open(img_path).convert("L")

        # Convert grayscale -> 3 channels
        # This allows use of pretrained ResNet models (which expect 3 channels)
        img = img.convert("RGB")

        label = int(row["label"])

        # Apply transform pipeline (resize, normalization, augmentation, etc.)
        if self.transform is not None:
            img = self.transform(img)

        return img, label
