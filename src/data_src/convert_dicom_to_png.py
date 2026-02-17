import argparse
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Normalize an image (often 12–16 bit range) to uint8 [0, 255].

    Steps:
    - Convert to float32
    - Apply per-image min–max normalization
    - Scale to [0, 255]
    - Convert to uint8

    This produces consistent PNG files while preserving relative contrast.
    Model normalization (mean/std) should be applied later in dataset.py.
    """
    image = image.astype(np.float32)

    min_val = float(np.min(image))
    max_val = float(np.max(image))

    if max_val == min_val:
        # Uniform image -> return mid-gray to avoid division by zero
        return np.zeros_like(image, dtype=np.uint8) + 128

    image = (image - min_val) / (max_val - min_val)
    image = (image * 255.0).clip(0, 255)

    return image.astype(np.uint8)


def dicom_to_corrected_pixels(ds: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Extract pixel data from DICOM and apply necessary intensity corrections.

    Corrections applied:
    - RescaleSlope / RescaleIntercept (if present)
    - MONOCHROME1 inversion handling

    Returns corrected float32 pixel array.
    """
    img = ds.pixel_array.astype(np.float32)

    # Apply rescale slope and intercept if available
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = img * slope + intercept

    # Handle MONOCHROME1 inversion
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = np.max(img) - img

    return img


def convert_single_dicom(dicom_path: Path, png_path: Path) -> bool:
    """
    Convert a single DICOM file to PNG (grayscale).

    Returns:
        True if successful
        False if conversion failed
    """
    try:
        ds = pydicom.dcmread(str(dicom_path))

        # Apply DICOM corrections
        corrected_pixels = dicom_to_corrected_pixels(ds)

        # Normalize to uint8
        img_uint8 = normalize_to_uint8(corrected_pixels)

        # Save as PNG
        img = Image.fromarray(img_uint8, mode="L")
        png_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(png_path))

        return True

    except Exception as e:
        print(f"[WARNING] Failed to convert {dicom_path}: {e}")
        return False


def convert_folder(
    input_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
    log_bad_dicoms_path: Path | None = None
) -> None:
    """
    Recursively convert all DICOM files from input_dir to PNG format
    in output_dir while preserving directory structure.

    Optionally logs corrupted or unreadable DICOM files.
    """
    dicom_files = list(input_dir.rglob("*.dcm"))

    if not dicom_files:
        print(f"[INFO] No DICOM files found in {input_dir}")
        return

    print(f"[INFO] Found {len(dicom_files)} DICOM files.")
    print(f"[INFO] Converting to PNG in {output_dir}")

    bad_dicoms: list[str] = []
    n_converted = 0
    n_skipped = 0

    for dcm_path in tqdm(dicom_files, desc="Converting DICOM to PNG"):
        rel_path = dcm_path.relative_to(input_dir)
        png_rel_path = rel_path.with_suffix(".png")
        png_path = output_dir / png_rel_path

        if png_path.exists() and not overwrite:
            n_skipped += 1
            continue

        success = convert_single_dicom(dcm_path, png_path)

        if success:
            n_converted += 1
        else:
            bad_dicoms.append(str(dcm_path))

    print("[INFO] Conversion finished.")
    print(f"[INFO] Converted: {n_converted}")
    print(f"[INFO] Skipped (already exists): {n_skipped}")
    print(f"[INFO] Failed: {len(bad_dicoms)}")

    # Save list of failed DICOM files if requested
    if log_bad_dicoms_path is not None:
        log_bad_dicoms_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_bad_dicoms_path, "w", encoding="utf-8") as f:
            for p in bad_dicoms:
                f.write(p + "\n")
        print(f"[INFO] Failed DICOM list saved to: {log_bad_dicoms_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RSNA DICOM chest X-rays to PNG."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing DICOM files (e.g., data/raw/train).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where PNG images will be saved (e.g., data/png/train).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files.",
    )
    parser.add_argument(
        "--log_bad_dicoms",
        type=str,
        default=None,
        help="Optional path to save list of corrupted DICOM files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    log_bad_dicoms_path = Path(args.log_bad_dicoms) if args.log_bad_dicoms else None

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    convert_folder(
        input_dir,
        output_dir,
        overwrite=args.overwrite,
        log_bad_dicoms_path=log_bad_dicoms_path
    )


if __name__ == "__main__":
    main()

"""
# Convert training images (used for splits and training)
python src\data_src\convert_dicom_to_png.py --input_dir data\raw\train --output_dir data\png\train --log_bad_dicoms results\bad_dicoms_train.txt
"""
