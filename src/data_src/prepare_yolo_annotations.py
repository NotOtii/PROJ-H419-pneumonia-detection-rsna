import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


def rsna_to_yolo(
    labels_csv: Path,
    images_dir: Path,
    output_dir: Path,
    create_empty_for_negatives: bool = True,
) -> None:
    """
    Convert RSNA bounding-box annotations to YOLO format.

    For each patientId:
        - if Target == 1 -> one or more pneumonia bounding boxes
        - if Target == 0 -> no pneumonia (negative image)

    This function creates one .txt file per image with the YOLO labels:
        class x_center y_center width height   (all normalized to [0, 1])

    class = 0  (single class: pneumonia)
    """
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading RSNA labels from {labels_csv}")
    df = pd.read_csv(labels_csv)

    required_cols = {"patientId", "Target", "x", "y", "width", "height"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns: {required_cols}, found: {set(df.columns)}"
        )

    # Group annotations by patientId
    grouped = df.groupby("patientId")

    num_pos = 0
    num_neg = 0
    num_skipped = 0

    for patient_id, group in tqdm(grouped, desc="Converting RSNA -> YOLO"):
        # Corresponding image path (after DICOM -> PNG conversion)
        img_path = images_dir / f"{patient_id}.png"

        if not img_path.exists():
            # Image not found (maybe not converted) -> skip
            num_skipped += 1
            continue

        # Open image to get width/height
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        # Output YOLO label file
        label_path = output_dir / f"{patient_id}.txt"

        # If Target = 0 for all rows -> negative example
        if group["Target"].max() == 0:
            num_neg += 1
            if create_empty_for_negatives:
                # Create an empty .txt file (no objects), YOLO interprets as background image
                label_path.touch()
            # Otherwise: skip creating any label file
            continue

        # If we are here, at least one box has Target = 1 (pneumonia)
        num_pos += 1

        yolo_lines = []

        # Keep only rows with Target = 1 (i.e., actual boxes)
        for _, row in group[group["Target"] == 1].iterrows():
            x = float(row["x"])
            y = float(row["y"])
            w = float(row["width"])
            h = float(row["height"])

            # Convert from (x, y, width, height) in pixels to YOLO format (normalized)
            x_center = (x + w / 2.0) / img_w
            y_center = (y + h / 2.0) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            # Single-class problem -> class_id = 0
            line = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            yolo_lines.append(line)

        with label_path.open("w") as f:
            for line in yolo_lines:
                f.write(line + "\n")

    print(f"[INFO] Positive images with boxes: {num_pos}")
    print(f"[INFO] Negative images (no pneumonia): {num_neg}")
    print(f"[INFO] Skipped images (no PNG found): {num_skipped}")
    print(f"[INFO] YOLO label files written to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert RSNA Pneumonia Detection Challenge annotations "
            "into YOLO label files."
        )
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        required=True,
        help="Path to RSNA labels CSV (e.g., data/raw/stage_1_train_labels.csv).",
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
        default="data/yolo/labels_all",
        help="Folder where YOLO .txt label files will be saved.",
    )
    parser.add_argument(
        "--no_empty_negatives",
        action="store_true",
        help="If set, do not create empty label files for negative images.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    labels_csv = Path(args.labels_csv)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    rsna_to_yolo(
        labels_csv=labels_csv,
        images_dir=images_dir,
        output_dir=output_dir,
        create_empty_for_negatives=not args.no_empty_negatives,
    )


if __name__ == "__main__":
    main()


"""
Command to run in the terminal to generate YOLO annotations:
------------------------------------------------------------
Run this from the root of the project (PROJ-H419-Pneumonia):

python src/data_src/prepare_yolo_annotations.py --labels_csv data/raw/stage_1_train_labels.csv --images_dir data/png/train --output_dir data/yolo/labels_all


------------------------------------------------------------
SCRIPT EXPLANATION: prepare_yolo_annotations.py
------------------------------------------------------------

Goal
----
This script converts the RSNA Pneumonia Detection Challenge annotations
into YOLO-compatible .txt label files.

For each X-ray image:
    - If Target = 1 -> one or more pneumonia bounding boxes
    - If Target = 0 -> negative image (no pneumonia)

We create one YOLO label file per image, named:
    <patientId>.txt

Each line in the file has the format:
    class_id x_center y_center width height

Where:
    - class_id = 0  (single class: pneumonia)
    - coordinates are normalized to [0, 1]
    - (x_center, y_center) is the box center
    - (width, height) is the box size

How it works
------------

1. Read the RSNA CSV
   - The CSV contains columns:
        'patientId', 'Target', 'x', 'y', 'width', 'height'
   - There can be multiple rows per patientId.

2. Group by patientId
   - For each patientId, we load the corresponding PNG image:
        images_dir / "<patientId>.png"
   - We get its width and height using PIL.

3. Handle negatives and positives
   - If all Target = 0 for that patientId:
        • the image is considered normal (no pneumonia).
        • we optionally create an empty .txt file (no objects).
   - If at least one row has Target = 1:
        • each row with Target = 1 is converted into a YOLO bounding box.

4. Convert RSNA boxes to YOLO format
   - RSNA provides (x, y, width, height) in pixels,
     with (x, y) = top-left corner.
   - YOLO expects normalized:
        x_center = (x + width/2) / image_width
        y_center = (y + height/2) / image_height
        w_norm = width / image_width
        h_norm = height / image_height

5. Save YOLO labels
   - One file per image in output_dir.
   - These files will later be used by another script to create
     the final YOLO dataset structure:

        data/yolo/images/train/
        data/yolo/images/val/
        data/yolo/labels/train/
        data/yolo/labels/val/

     using the train/val splits.

Note on splits
--------------
This script does NOT use train/val/test splits directly.
It simply generates YOLO annotations for all images in images_dir.

The split files (data/splits/train.txt, val.txt, test.txt) will be used
later (e.g., by a `prepare_yolo_dataset.py` script) to distribute images
and labels into YOLO train/val folders.
------------------------------------------------------------
"""
