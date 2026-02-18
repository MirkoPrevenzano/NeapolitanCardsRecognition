"""
Rinomina immagini in setNotAugmentation/train
Formato: <seme>_<valore>_<xxxx>.<ext>
- seme: secondo token della cartella (es. 'A','B','C','D')
- valore: primo token della cartella (es. '1','2',...,'10')
- xxxx: progressivo a 4 cifre per quella classe
Genera anche un file CSV `rename_mapping.csv` con mapping old->new
Esempio di cartella: "1 A/WhatsApp-Image-...jpg" -> "A_1_0001.jpg"

Usage:
  python3 utilScript/rename_images.py --base-dir <path> [--output-csv <path>]
"""
from pathlib import Path
import csv
import argparse

EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

def rename_images(base_dir, output_csv):
    BASE = Path(base_dir)
    mapping = []

    for class_dir in sorted(BASE.iterdir()):
        if not class_dir.is_dir():
            continue
        # Expect folder names like "1 A" or "10 D"
        parts = class_dir.name.split()
        if len(parts) >= 2:
            valore = parts[1]
            seme = parts[0]
        else:
            # fallback: use full folder name
            valore = parts[0]
            seme = parts[0]

        # prepare counter for this class
        counter = 1
        # iterate files sorted
        for img in sorted(class_dir.iterdir()):
            if not img.is_file():
                continue
            if img.suffix.lower() not in EXTS:
                continue
            ext = img.suffix.lower()
            new_name = f"{seme}_{valore}_{counter:04d}{ext}"
            new_path = class_dir / new_name
            # if exists, find next free counter
            while new_path.exists():
                counter += 1
                new_name = f"{seme}_{valore}_{counter:04d}{ext}"
                new_path = class_dir / new_name
            # perform rename
            img.rename(new_path)
            mapping.append((str(img), str(new_path)))
            counter += 1

    # write mapping csv
    csv_path = Path(output_csv)
    with csv_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["old_path", "new_path"])
        for old, new in mapping:
            writer.writerow([old, new])

    print(f"Rinominati {len(mapping)} file. Mapping salvato in {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rinomina immagini con formato seme_valore_xxxx")
    parser.add_argument("--base-dir", required=True, help="Cartella base con sottocartelle di classi")
    parser.add_argument("--output-csv", default="rename_mapping.csv", help="File CSV output mapping")
    args = parser.parse_args()
    
    rename_images(args.base_dir, args.output_csv)

