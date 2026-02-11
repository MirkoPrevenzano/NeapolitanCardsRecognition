"""
Script di Data Augmentation per il dataset di carte napoletane.
Applica 8 trasformazioni a ogni immagine nella cartella train,
salvando i risultati nella stessa cartella della classe originale.

Trasformazioni applicate:
  1. Flip verticale
  2. Rotazione +90° (bordi neri)
  3. Rotazione -90° (bordi neri)
  4. Greyscale (convertita in 3 canali per compatibilità)
  5. Variazione di saturazione
  6. Variazione di luminosità
  7. Aggiunta di rumore gaussiano
  8. Rotazione +15° (bordi neri)
  9. Rotazione -15° (bordi neri)
  10. Leggera sfocatura (Gaussian blur)

Usage:
  python3 utilScript/augmentation.py --input-dir <path> [--output-dir <path>]
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path


def flip_vertical(img):
    """Flip verticale dell'immagine."""
    return cv2.flip(img, 0)


def rotate(img, angle):
    """Rotazione di un angolo arbitrario con bordi neri, stessa dimensione."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderValue=(0, 0, 0))


def to_greyscale(img):
    """Conversione in scala di grigi (3 canali per compatibilità)."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def change_saturation(img, factor=1.5):
    """Variazione della saturazione (factor > 1 = più saturo)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def change_brightness(img, factor=1.3):
    """Variazione della luminosità (factor > 1 = più luminoso)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def add_noise(img, sigma=25):
    """Aggiunta di rumore gaussiano."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def gaussian_blur(img, ksize=5):
    """Leggera sfocatura gaussiana."""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


# Dizionario con tutte le trasformazioni: nome_suffisso -> funzione
AUGMENTATIONS = {
    "flip_v":       lambda img: flip_vertical(img),
    "rot_p90":      lambda img: rotate(img, 90),
    "rot_m90":      lambda img: rotate(img, -90),
    "greyscale":    lambda img: to_greyscale(img),
    "saturazione":  lambda img: change_saturation(img, factor=1.5),
    "luminosita":   lambda img: change_brightness(img, factor=1.3),
    "noise":        lambda img: add_noise(img, sigma=25),
    "rot_p15":      lambda img: rotate(img, 15),
    "rot_m15":      lambda img: rotate(img, -15),
    "blur":         lambda img: gaussian_blur(img, ksize=5),
}


def process_directory(input_dir):
    """Applica tutte le augmentation a ogni immagine nel dataset."""

    input_path = Path(input_dir)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    # Raccolta di tutte le immagini originali
    image_files = []
    for class_dir in sorted(input_path.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in extensions:
                image_files.append(img_file)

    total = len(image_files)
    print(f"Trovate {total} immagini in {input_dir}")
    print(f"Augmentations da applicare: {len(AUGMENTATIONS)} per immagine")
    print(f"Totale immagini finali: {total} originali + {total * len(AUGMENTATIONS)} augmentate "
          f"= {total * (1 + len(AUGMENTATIONS))}")
    print("-" * 60)

    created = 0
    for idx, img_path in enumerate(image_files, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [SKIP] Impossibile leggere: {img_path.name}")
            continue

        stem = img_path.stem      # nome senza estensione
        suffix = img_path.suffix  # estensione (.jpg)
        class_dir = img_path.parent

        for aug_name, aug_fn in AUGMENTATIONS.items():
            out_name = f"{stem}_aug_{aug_name}{suffix}"
            out_path = class_dir / out_name

            augmented = aug_fn(img)
            cv2.imwrite(str(out_path), augmented)
            created += 1

        print(f"  [{idx}/{total}] {class_dir.name}/{img_path.name} -> {len(AUGMENTATIONS)} augmentazioni")

    print("-" * 60)
    print(f"Completato! {created} immagini augmentate create.")
    print(f"Totale immagini nella cartella: {total + created}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augmentation dataset con 10 trasformazioni")
    parser.add_argument("--input-dir", required=True, help="Cartella input con immagini")
    args = parser.parse_args()
    
    process_directory(args.input_dir)

