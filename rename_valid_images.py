#!/usr/bin/env python3
import os
import shutil
import csv

root = 'setNotAugmentation/valid'
mapping_path = 'rename_valid_mapping.csv'

rows = []
for class_dir in sorted(os.listdir(root)):
    class_path = os.path.join(root, class_dir)
    if not os.path.isdir(class_path):
        continue
    parts = class_dir.split()
    if len(parts) >= 2:
        numero = parts[0]
        seme = parts[1]
    elif len(parts) == 1:
        # fallback: try to parse like '1_A' or '1A'
        p = parts[0]
        if '_' in p:
            numero, seme = p.split('_', 1)
        else:
            # unknown, use whole as seme and numero 0
            numero = p
            seme = 'X'
    else:
        numero = '0'
        seme = 'X'

    files = sorted([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
    counter = 1
    for fname in files:
        old_path = os.path.join(class_path, fname)
        name, ext = os.path.splitext(fname)
        new_fname = f"{numero}_{seme}_{counter:04d}{ext}"
        new_path = os.path.join(class_path, new_fname)
        # avoid overwriting existing file
        if os.path.exists(new_path):
            # if exists, increment counter until unique
            while os.path.exists(new_path):
                counter += 1
                new_fname = f"{numero}_{seme}_{counter:04d}{ext}"
                new_path = os.path.join(class_path, new_fname)
        try:
            shutil.move(old_path, new_path)
            rows.append((old_path, new_path))
            counter += 1
        except Exception as e:
            print(f"Failed to rename {old_path}: {e}")

# write mapping csv
with open(mapping_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['old_path', 'new_path'])
    for r in rows:
        w.writerow(r)

print(f"Renamed {len(rows)} files. Mapping saved to {mapping_path}.")
