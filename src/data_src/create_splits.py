import argparse
from pathlib import Path

import pandas as pd

import os
print("RUNNING:", os.path.abspath(__file__))


def stratified_split(
    df: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a stratified train/val/test split on a binary-labeled dataframe.

    df must contain:
        - image_path
        - label (0 or 1)

    The split preserves the class distribution across subsets.
    """
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    # Separate classes
    df_pos = df[df["label"] == 1].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_neg = df[df["label"] == 0].sample(frac=1.0, random_state=seed).reset_index(drop=True)

    def split_class(df_class: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n_total = len(df_class)
        n_val = int(n_total * val_ratio)
        n_test = int(n_total * test_ratio)
        n_train = n_total - n_val - n_test

        train = df_class.iloc[:n_train]
        val = df_class.iloc[n_train:n_train + n_val]
        test = df_class.iloc[n_train + n_val:]
        return train, val, test

    pos_train, pos_val, pos_test = split_class(df_pos)
    neg_train, neg_val, neg_test = split_class(df_neg)

    # Combine and shuffle within each split
    train_df = pd.concat([pos_train, neg_train]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat([pos_val, neg_val]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat([pos_test, neg_test]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return train_df, val_df, test_df


def write_split_file(df: pd.DataFrame, output_file: Path) -> None:
    """
    Write image paths to a .txt file (one path per line).
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        for p in df["image_path"].tolist():
            # Force forward slashes for portability
            f.write(str(Path(p).as_posix()) + "\n")


def summarize_split(name: str, df: pd.DataFrame) -> str:
    """
    Return a readable summary line for a split.
    """
    total = len(df)
    positives = int(df["label"].sum())
    negatives = total - positives
    ratio = positives / total if total > 0 else 0.0
    return f"{name}: total={total}, pneumonia={positives}, normal={negatives}, pos_ratio={ratio:.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create stratified train/val/test splits using classification_labels.csv."
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        default="data/classification_labels.csv",
        help="CSV containing columns: image_path,label (default: data/classification_labels.csv).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/splits",
        help="Folder where train.txt, val.txt, test.txt will be saved.",
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

    labels_csv = Path(args.labels_csv)
    output_dir = Path(args.output_dir)

    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    df = pd.read_csv(labels_csv)

    if "image_path" not in df.columns or "label" not in df.columns:
        raise ValueError("labels_csv must contain columns: image_path,label")

    # Ensure label is int
    df["label"] = df["label"].astype(int)

    # Stratified split
    train_df, val_df, test_df = stratified_split(
        df, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )

    print("[INFO] Stratified splits created:")
    print("[INFO] " + summarize_split("TRAIN", train_df))
    print("[INFO] " + summarize_split("VAL", val_df))
    print("[INFO] " + summarize_split("TEST", test_df))

    # Save split txt files (paths only)
    write_split_file(train_df, output_dir / "train.txt")
    write_split_file(val_df, output_dir / "val.txt")
    write_split_file(test_df, output_dir / "test.txt")

    # Save a summary report (useful for your write-up)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "splits_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(summarize_split("TRAIN", train_df) + "\n")
        f.write(summarize_split("VAL", val_df) + "\n")
        f.write(summarize_split("TEST", test_df) + "\n")

    print(f"[INFO] Split files saved in {output_dir}")
    print(f"[INFO] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()


"""
How to run:
-----------
From the project root:

python -c "import runpy, sys; sys.argv=['create_splits.py','--labels_csv','data/classification_labels.csv','--output_dir','data/splits','--val_ratio','0.15','--test_ratio','0.15','--seed','42']; runpy.run_path('src/data_src/create_splits.py', run_name='__main__')"

What this produces:
-------------------
- data/splits/train.txt
- data/splits/val.txt
- data/splits/test.txt
- data/splits/splits_summary.txt

Each .txt file contains one image path per line.

The split is stratified, so the pneumonia/normal ratio is preserved across train/val/test.
"""
