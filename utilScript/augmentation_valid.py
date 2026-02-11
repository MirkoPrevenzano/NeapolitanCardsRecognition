"""
Script di Data Augmentation per il validation set.
Applica solo 5 trasformazioni geometriche a ogni immagine in setNotAugmentation/valid.

Trasformazioni:
  1. Flip verticale
  2. Rotazione +90°
  3. Rotazione -90°
  4. Rotazione +15°
  5. Rotazione -15°

Usage:
  python3 utilScript/augmentation_valid.py --input-dir <path>
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def flip_vertical(img):
    return cv2.flip(img, 0)


def rotate(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderValue=(0, 0, 0))


AUGMENTATIONS = {
    "flip_v":  lambda img: flip_vertical(img),
    "rot_p90": lambda img: rotate(img, 90),
    "rot_m90": lambda img: rotate(img, -90),
    "rot_p15": lambda img: rotate(img, 15),
    "rot_m15": lambda img: rotate(img, -15),
}


def process_directory(input_dir):
    input_path = Path(input_dir)
    image_files = []
    for class_dir in sorted(input_path.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in EXTS:
                image_files.append(img_file)

    total = len(image_files)
    print(f"Trovate {total} immagini in {input_dir}")
    print(f"Augmentations: {len(AUGMENTATIONS)} per immagine")
    print(f"Totale finale: {total} originali + {total * len(AUGMENTATIONS)} augmentate = {total * (1 + len(AUGMENTATIONS))}")
    print("-" * 60)

    created = 0
    for idx, img_path in enumerate(image_files, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [SKIP] {img_path.name}")
            continue

        stem = img_path.stem
        suffix = img_path.suffix
        class_dir = img_path.parent

        for aug_name, aug_fn in AUGMENTATIONS.items():
            out_name = f"{stem}_aug_{aug_name}{suffix}"
            out_path = class_dir / out_name
            cv2.imwrite(str(out_path), aug_fn(img))
            created += 1

        print(f"  [{idx}/{total}] {class_dir.name}/{img_path.name} -> {len(AUGMENTATIONS)} aug")

    print("-" * 60)
    print(f"Completato! {created} immagini augmentate create.")
    print(f"Totale immagini: {total + created}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augmentation validation set (5 trasformazioni geometriche)")
    parser.add_argument("--input-dir", required=True, help="Cartella input con immagini")
    args = parser.parse_args()
    
    process_directory(args.input_dir)
