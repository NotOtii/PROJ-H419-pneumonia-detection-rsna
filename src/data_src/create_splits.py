import argparse
import random
from pathlib import Path
from typing import List, Tuple


def list_images(images_dir: Path) -> list[Path]:
    """
    Returns the list of all PNG images inside images_dir (recursive).
    """
    return sorted(images_dir.rglob("*.png"))


def split_list(
    items: List[Path], val_ratio: float, test_ratio: float, seed: int = 42
) -> Tuple[list[Path], list[Path], list[Path]]:
    """
    Splits a list of images into three subsets: train / val / test.
    Ratios must sum to < 1.0.
    """
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    # Shuffle for randomness (seed ensures reproducibility)
    random.Random(seed).shuffle(items)

    n_total = len(items)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test

    train_items = items[:n_train]
    val_items = items[n_train : n_train + n_val]
    test_items = items[n_train + n_val :]

    return train_items, val_items, test_items


def write_split_file(paths: list[Path], output_file: Path) -> None:
    """
    Writes the list of image paths into a text file.
    Paths are written as-is (e.g., 'data/png/train/img.png'),
    which is suitable for training scripts.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w") as f:
        for p in paths:
            # Force '/' as path separator (works on all OS)
            f.write(str(p.as_posix()) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits from PNG images."
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Folder containing PNG images (e.g., data/png/train).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/splits",
        help="Folder where split files (train.txt, val.txt, test.txt) will be saved.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Proportion of images for validation (default: 0.15).",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Proportion of images for testing (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")

    images = list_images(images_dir)

    if not images:
        raise RuntimeError(f"No PNG images found in {images_dir}")

    print(f"[INFO] Found {len(images)} images in {images_dir}")

    # Create splits
    train_imgs, val_imgs, test_imgs = split_list(
        images, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )

    print(
        f"[INFO] Split into: "
        f"{len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test images."
    )

    # Save text files
    train_file = output_dir / "train.txt"
    val_file = output_dir / "val.txt"
    test_file = output_dir / "test.txt"

    write_split_file(train_imgs, train_file)
    write_split_file(val_imgs, val_file)
    write_split_file(test_imgs, test_file)

    print(f"[INFO] Splits saved in {output_dir}")


if __name__ == "__main__":
    main()


"""
Commands to generate the train/val/test splits:
-----------------------------------------------
Run this from the root of the project:

python src/data_src/create_splits.py --images_dir data/png/train --output_dir data/splits --val_ratio 0.15 --test_ratio 0.15


------------------------------------------------------------
SCRIPT EXPLANATION: create_splits.py
------------------------------------------------------------

This script creates three image subsets:
    - TRAIN: used to train the model
    - VALIDATION: used to tune hyperparameters and detect overfitting
    - TEST: used to evaluate final model performance

Why do we create splits?
------------------------
A model cannot be evaluated on the same images it was trained on.
Otherwise, it would simply memorize the dataset.

Data splits ensure:
    • fair evaluation
    • no data leakage
    • reliable generalization performance

How the script works
--------------------

1. list_images(images_dir)
   - Scans the folder recursively and finds all PNG files.
   - Returns them sorted.

2. split_list(items, val_ratio, test_ratio, seed)
   - Randomly shuffles the list of images.
   - Uses the provided ratios to divide them into:
        train / validation / test.
   - The seed ensures reproducibility.

3. write_split_file(paths, output_file)
   - Writes image paths to .txt files.
   - Paths are saved as strings (e.g. "data/png/train/001.png").

4. main()
   - Parses arguments.
   - Verifies inputs.
   - Loads all images.
   - Generates splits.
   - Saves three files:
        data/splits/train.txt
        data/splits/val.txt
        data/splits/test.txt

These files will later be used by training scripts 
(e.g., PyTorch DataLoader, YOLO training, etc.).
------------------------------------------------------------
"""
