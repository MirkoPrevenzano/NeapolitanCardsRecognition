#!/usr/bin/env python3
"""
Script per convertire immagini orizzontali in verticali con formato 3:4
Sovrascrive direttamente i file originali.

Usage:
  python3 utilScript/convert_to_portrait_4_3.py --path foto/mazzoFabrizio3
"""
import argparse
from pathlib import Path

try:
    from PIL import Image, ImageOps
except ImportError:
    print("ERRORE: Pillow non installato. Esegui: pip install Pillow")
    exit(1)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}
TARGET_RATIO = 3.0 / 4.0  # larghezza / altezza per verticale 3:4


def center_crop_to_ratio(img: Image.Image, target_ratio: float) -> Image.Image:
    """Ritaglia l'immagine al centro per rispettare il ratio target"""
    w, h = img.size
    cur_ratio = w / h
    
    if abs(cur_ratio - target_ratio) < 0.01:  # già nel formato giusto
        return img
    
    if cur_ratio > target_ratio:
        # troppo larga -> ritaglia larghezza
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    else:
        # troppo alta -> ritaglia altezza
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        return img.crop((0, top, w, top + new_h))


def process_image(path: Path) -> bool:
    """
    Processa un'immagine:
    - Applica orientamento EXIF
    - Se orizzontale, ruota di 90° per renderla verticale
    - Ritaglia al centro per formato 3:4
    - Sovrascrive il file originale
    
    Returns: True se modificata, False altrimenti
    """
    try:
        with Image.open(path) as img:
            # Applica orientamento EXIF
            img = ImageOps.exif_transpose(img)
            original_size = img.size
            w, h = img.size
            
            # Se orizzontale (larghezza > altezza), ruota di 90°
            if w > h:
                img = img.transpose(Image.Transpose.ROTATE_90)
                print(f"  ↻ Ruotata: {path.name} ({original_size} → {img.size})")
            else:
                print(f"  ✓ Già verticale: {path.name} ({img.size})")
            
            # Ritaglia al formato 3:4 (verticale)
            img_cropped = center_crop_to_ratio(img, TARGET_RATIO)
            
            if img_cropped.size != img.size:
                print(f"    ✂ Ritagliata: {img.size} → {img_cropped.size}")
            
            # Salva sovrascrivendo l'originale
            img_cropped.save(path, quality=95, optimize=True)
            return True
            
    except Exception as e:
        print(f"  ✗ ERRORE con {path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Converte immagini orizzontali in verticali (3:4) sovrascrivendole"
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Cartella con le immagini da processare"
    )
    args = parser.parse_args()
    
    folder = Path(args.path)
    if not folder.exists():
        print(f"ERRORE: Cartella non trovata: {folder}")
        return
    
    # Trova tutte le immagini
    images = [
        p for p in folder.rglob("*") 
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]
    
    if not images:
        print(f"Nessuna immagine trovata in {folder}")
        return
    
    print(f"Trovate {len(images)} immagini in {folder}")
    print(f"Formato target: 3:4 (verticale)\n")
    
    # Processa ogni immagine
    processed = 0
    for img_path in images:
        if process_image(img_path):
            processed += 1
    
    print(f"\n✓ Completato: {processed}/{len(images)} immagini processate")


if __name__ == "__main__":
    main()
