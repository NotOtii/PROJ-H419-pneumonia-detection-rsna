import os
import random
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import pydicom

# --- chemins à adapter si besoin, mais je garde ta logique actuelle ---
CSV_PATH   = r"C:\\Users\\houta\\Desktop\\MA2\\PROJ-H419-Pneumonia\\data\\raw\\stage_1_train_labels.csv"
DICOM_DIR  = r"C:\\Users\\houta\\Desktop\\MA2\\PROJ-H419-Pneumonia\\data\\raw\\train"
PNG_DIR    = r"C:\\Users\\houta\\Desktop\\MA2\\PROJ-H419-Pneumonia\\data\\png\\train"

# Charger le CSV
df = pd.read_csv(CSV_PATH)
print("Colonnes CSV :", df.columns.tolist())

# On prend un patientId avec Target = 1 (pneumonie)
patient_ids = df[df["Target"] == 1]["patientId"].drop_duplicates().tolist()
patient_id = random.choice(patient_ids)
print("patientId choisi :", patient_id)

# Chemins des fichiers
dicom_path = os.path.join(DICOM_DIR, f"{patient_id}.dcm")
png_path   = os.path.join(PNG_DIR,   f"{patient_id}.png")

if not os.path.exists(dicom_path):
    raise FileNotFoundError(f"DICOM introuvable : {dicom_path}")
if not os.path.exists(png_path):
    raise FileNotFoundError(f"PNG introuvable : {png_path}")

# ---------- 1) Charger DICOM ----------
ds = pydicom.dcmread(dicom_path)
img_dicom = ds.pixel_array

# Normaliser en 0–255 et convertir en RGB pour dessiner
img_dicom_norm = cv2.normalize(img_dicom, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
img_dicom_rgb  = cv2.cvtColor(img_dicom_norm, cv2.COLOR_GRAY2RGB)

h_d, w_d = img_dicom_rgb.shape[:2]
print("Taille DICOM :", w_d, "x", h_d)

# ---------- 2) Charger PNG ----------
img_png = cv2.imread(png_path)
img_png = cv2.cvtColor(img_png, cv2.COLOR_BGR2RGB)
h_p, w_p = img_png.shape[:2]
print("Taille PNG   :", w_p, "x", h_p)

# ---------- 3) Récupérer les boxes de CE patient ----------
subset = df[(df["patientId"] == patient_id) & (df["Target"] == 1)]
print("Nombre de boxes pour cette image :", len(subset))
print(subset[["x", "y", "width", "height"]])

# ---------- 4) Dessiner sur DICOM (coords originales) ----------
img_dicom_draw = img_dicom_rgb.copy()

for _, r in subset.iterrows():
    x = int(r["x"])
    y = int(r["y"])
    w_box = int(r["width"])
    h_box = int(r["height"])

    cv2.rectangle(
        img_dicom_draw,
        (x, y),
        (x + w_box, y + h_box),
        (255, 0, 0),
        3,
    )

# ---------- 5) Dessiner sur PNG (coords rescalées si ta conversion a changé la taille) ----------
img_png_draw = img_png.copy()

scale_x = w_p / w_d
scale_y = h_p / h_d
print("scale_x, scale_y :", scale_x, scale_y)

for _, r in subset.iterrows():
    x = int(r["x"] * scale_x)
    y = int(r["y"] * scale_y)
    w_box = int(r["width"] * scale_x)
    h_box = int(r["height"] * scale_y)

    cv2.rectangle(
        img_png_draw,
        (x, y),
        (x + w_box, y + h_box),
        (255, 0, 0),
        3,
    )

# ---------- 6) Affichage côte à côte ----------
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("DICOM + boxes (coords originales)")
plt.imshow(img_dicom_draw)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("PNG + boxes (après conversion)")
plt.imshow(img_png_draw)
plt.axis("off")

plt.tight_layout()
plt.show()
