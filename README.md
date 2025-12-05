This folder stores all local copies of the RSNA Pneumonia dataset.

## Important
The dataset is **NOT** included in this repository because of:
- its large size
- licensing restrictions (Kaggle rules)
- medical data privacy

This folder is ignored by Git using the `.gitignore` file.

## Folder Structure

```
data/
  raw/        → Original DICOM files downloaded from Kaggle
  png/        → Converted PNG images (after preprocessing)
  splits/     → train/val/test text files
  yolo/       → YOLO-ready images + bounding-box annotations
```

Place the RSNA dataset files here before running any preprocessing or training scripts.
