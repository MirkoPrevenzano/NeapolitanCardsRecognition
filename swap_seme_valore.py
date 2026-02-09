"""
Script per rinominare i file da formato `seme_valore_xxxx.ext` a `valore_seme_xxxx.ext`.
Genera `/home/mirko/Scrivania/NN/rename_swap_mapping.csv` con old_path,new_path.
Funziona ricorsivamente su `setNotAugmentation/train`.
"""
from pathlib import Path
import csv
import re

BASE = Path("/home/mirko/Scrivania/NN/setNotAugmentation/train")
OUT_CSV = Path("/home/mirko/Scrivania/NN/rename_swap_mapping.csv")

pattern = re.compile(r'^(?P<seme>[A-Za-z]+)_(?P<valore>\d{1,2})_(?P<cnt>\d{4})(?P<ext>\.[A-Za-z0-9]+)$')

mapping = []
renamed = 0
skipped = 0

for class_dir in sorted(BASE.iterdir()):
    if not class_dir.is_dir():
        continue
    for f in sorted(class_dir.iterdir()):
        if not f.is_file():
            continue
        m = pattern.match(f.name)
        if not m:
            skipped += 1
            continue
        seme = m.group('seme')
        valore = m.group('valore')
        cnt = m.group('cnt')
        ext = m.group('ext')

        new_name = f"{valore}_{seme}_{cnt}{ext}"
        new_path = class_dir / new_name

        # Se il percorso destinazione esiste, evita collisione aggiungendo un suffisso
        if new_path.exists():
            # trova primo numero libero
            i = 1
            while True:
                candidate = class_dir / f"{valore}_{seme}_{cnt}_{i}{ext}"
                if not candidate.exists():
                    new_path = candidate
                    break
                i += 1

        # esegui rename
        old = str(f)
        f.rename(new_path)
        mapping.append((old, str(new_path)))
        renamed += 1

# scrivi CSV
with OUT_CSV.open('w', newline='') as csvf:
    writer = csv.writer(csvf)
    writer.writerow(["old_path", "new_path"])
    for old, new in mapping:
        writer.writerow([old, new])

print(f"Rinominati: {renamed}, saltati (non matching): {skipped}")
print(f"Mapping salvato in: {OUT_CSV}")
