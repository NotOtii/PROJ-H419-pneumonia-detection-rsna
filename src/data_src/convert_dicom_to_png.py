import argparse
import os
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Normalise une image (généralement 16 bits) en 0–255 uint8.
    """
    image = image.astype(np.float32)

    min_val = np.min(image)
    max_val = np.max(image)

    if max_val == min_val:
        # Image uniforme -> retourne un gris moyen
        return np.zeros_like(image, dtype=np.uint8) + 128

    image = (image - min_val) / (max_val - min_val)  # 0–1
    image = (image * 255.0).clip(0, 255)

    return image.astype(np.uint8)


def convert_single_dicom(dicom_path: Path, png_path: Path) -> None:
    """
    Convertit un fichier DICOM vers PNG (grayscale).
    """
    try:
        ds = pydicom.dcmread(str(dicom_path))
        pixel_array = ds.pixel_array

        # Normalisation vers 0–255
        img_uint8 = normalize_to_uint8(pixel_array)

        # Convertir en image PIL (mode 'L' = grayscale 8 bits)
        img = Image.fromarray(img_uint8, mode="L")

        png_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(png_path))
    except Exception as e:
        print(f"[WARNING] Failed to convert {dicom_path}: {e}")


def convert_folder(input_dir: Path, output_dir: Path, overwrite: bool = False) -> None:
    """
    Parcourt récursivement input_dir, trouve tous les .dcm et les convertit en .png
    dans output_dir en gardant la même structure de sous-dossiers.
    """
    dicom_files = list(input_dir.rglob("*.dcm"))

    if not dicom_files:
        print(f"[INFO] No DICOM files found in {input_dir}")
        return

    print(f"[INFO] Found {len(dicom_files)} DICOM files.")
    print(f"[INFO] Converting to PNG in {output_dir}")

    for dcm_path in tqdm(dicom_files, desc="Converting DICOM to PNG"):
        # Chemin relatif par rapport à input_dir
        rel_path = dcm_path.relative_to(input_dir)
        # Remplace l'extension .dcm -> .png
        png_rel_path = rel_path.with_suffix(".png")
        png_path = output_dir / png_rel_path

        if png_path.exists() and not overwrite:
            # On saute si l'image existe déjà
            continue

        convert_single_dicom(dcm_path, png_path)

    print("[INFO] Conversion finished.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RSNA DICOM chest X-rays to PNG."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root folder containing DICOM files (e.g. data/raw/train).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root folder where PNG images will be saved (e.g. data/png/train).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files if they already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    convert_folder(input_dir, output_dir, overwrite=args.overwrite)
 

if __name__ == "__main__":
    main()


"""
Commands to run in the terminal to convert train and test DICOM files to PNG:
python src/data_src/convert_dicom_to_png.py --input_dir data/raw/train --output_dir data/png/train
python src/data_src/convert_dicom_to_png.py --input_dir data/raw/test --output_dir data/png/test

------------------------------------------------------------
EXPLICATION DU SCRIPT : convert_dicom_to_png.py
------------------------------------------------------------

Objectif du script
------------------
Ce script convertit les radiographies thoraciques du format DICOM 
(format médical contenant l’image + des métadonnées) vers le format PNG.

Les fichiers DICOM ne peuvent pas être utilisés directement pour entraîner
des modèles de deep learning (ResNet, YOLO, etc.) car :
    • ils sont souvent codés en 12–16 bits
    • ils contiennent des tags médicaux inutiles pour le modèle
    • ils ne sont pas compatibles avec les DataLoader classiques

Convertir les images en PNG 8 bits normalisés est donc une étape essentielle
du prétraitement du dataset RSNA Pneumonia Detection.

------------------------------------------------------------
Détail du fonctionnement du script
------------------------------------------------------------

1. normalize_to_uint8(image)
   -----------------------------------
   - Les images DICOM sont souvent encodées en haute profondeur 
     (12—16 bits avec valeurs max pouvant dépasser 4095).
   - Cette fonction :
        * convertit en float32
        * calcule min et max
        * applique une normalisation min–max → [0, 255]
        * renvoie une image uint8 compatible avec PNG
   - Si l'image est totalement uniforme, on renvoie un gris constant (128)
     pour éviter une division par zéro.

2. convert_single_dicom(dicom_path, png_path)
   -------------------------------------------
   - Lit le fichier DICOM avec pydicom.dcmread()
   - Récupère l’image via ds.pixel_array
   - Normalise l’image 16 bits en 8 bits
   - Convertit en image PIL en niveaux de gris ("L")
   - Crée le dossier de sortie si nécessaire
   - Sauvegarde au format PNG
   - En cas d’erreur (DICOM corrompu → courant dans RSNA), 
     un warning est affiché sans interrompre le script.

3. convert_folder(input_dir, output_dir, overwrite)
   --------------------------------------------------
   - Parcourt récursivement input_dir pour trouver tous les .dcm
   - Conserve la même structure de dossiers dans output_dir
   - Convertit chaque image une par une avec une barre de progression tqdm
   - Si overwrite=False, les PNG déjà existants sont ignorés
   - À la fin : affiche le nombre total d’images converties

4. parse_args()
   --------------------------------------------------
   Permet d'exécuter le script depuis le terminal avec les arguments :
       --input_dir   dossier contenant les DICOM (ex: data/raw/train)
       --output_dir  dossier où stocker les PNG (ex: data/png/train)
       --overwrite   écrase les PNG existants si fourni

5. main()
   --------------------------------------------------
   - Vérifie que le dossier d’entrée existe
   - Lance convert_folder() avec les chemins fournis

------------------------------------------------------------
Pourquoi cette conversion est nécessaire ?
------------------------------------------------------------
Les modèles de deep learning ne fonctionnent pas avec le format DICOM.
Ils nécessitent :
    • des images RGB ou grayscale en uint8 (0–255)
    • un format simple (PNG)
    • une organisation propre pour le DataLoader

La conversion DICOM → PNG est donc indispensable avant :
    - la classification (ResNet)
    - la détection (YOLO)
    - toute visualisation
    - toute création de dataset custom

------------------------------------------------------------
Commandes à exécuter dans le terminal
------------------------------------------------------------

# Conversion du dossier TRAIN
python src/data_src/convert_dicom_to_png.py --input_dir data/raw/train --output_dir data/png/train

# Conversion du dossier TEST
python src/data_src/convert_dicom_to_png.py --input_dir data/raw/test --output_dir data/png/test

------------------------------------------------------------
"""
