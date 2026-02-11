# Utility Scripts - Documentazione

Tutti gli script in questa cartella accettano path come parametri da linea di comando.

## convert_to_portrait_4_3.py
Converte immagini orizzontali in verticali con formato 3:4.

```bash
python3 utilScript/convert_to_portrait_4_3.py --path foto/mazzoFabrizio3
```

**Parametri:**
- `--path`: Cartella con le immagini da processare (obbligatorio)

**Comportamento:**
- Ruota le immagini orizzontali di 90°
- Ritaglia al centro per rispettare il formato 3:4
- Sovrascrive i file originali

---

## augmentation.py
Applica 10 trasformazioni di data augmentation alle immagini.

```bash
python3 utilScript/augmentation.py --input-dir setNotAugmentation/train
```

**Parametri:**
- `--input-dir`: Cartella input con immagini organizzate in sottocartelle per classe (obbligatorio)

**Trasformazioni applicate:**
1. Flip verticale
2. Rotazione +90°
3. Rotazione -90°
4. Greyscale
5. Variazione saturazione
6. Variazione luminosità
7. Rumore gaussiano
8. Rotazione +15°
9. Rotazione -15°
10. Gaussian blur

---

## augmentation_valid.py
Applica 5 trasformazioni geometriche al validation set.

```bash
python3 utilScript/augmentation_valid.py --input-dir setNotAugmentation/valid
```

**Parametri:**
- `--input-dir`: Cartella input con immagini (obbligatorio)

**Trasformazioni:**
1. Flip verticale
2. Rotazione +90°
3. Rotazione -90°
4. Rotazione +15°
5. Rotazione -15°

---

## rename_images.py
Rinomina immagini con formato `<seme>_<valore>_<xxxx>.<ext>`.

```bash
python3 utilScript/rename_images.py --base-dir setNotAugmentation/train --output-csv rename_mapping.csv
```

**Parametri:**
- `--base-dir`: Cartella base con sottocartelle di classi (obbligatorio)
- `--output-csv`: File CSV output mapping (default: `rename_mapping.csv`)

**Esempio:**
- Cartella: `1 A/WhatsApp-Image-...jpg`
- Output: `1 A/A_1_0001.jpg`

---

## rename_valid_images.py
Rinomina immagini nel validation set con formato `<numero>_<seme>_<xxxx>.<ext>`.

```bash
python3 utilScript/rename_valid_images.py --root-dir setNotAugmentation/valid --mapping-csv rename_valid_mapping.csv
```

**Parametri:**
- `--root-dir`: Cartella root del validation set (obbligatorio)
- `--mapping-csv`: File CSV output mapping (default: `rename_valid_mapping.csv`)

---

## swap_seme_valore.py
Scambia l'ordine di seme e valore nei nomi file: `seme_valore_xxxx` → `valore_seme_xxxx`.

```bash
python3 utilScript/swap_seme_valore.py --base-dir setNotAugmentation/train --output-csv rename_swap_mapping.csv
```

**Parametri:**
- `--base-dir`: Cartella base con sottocartelle (obbligatorio)
- `--output-csv`: File CSV output mapping (default: `rename_swap_mapping.csv`)

**Esempio:**
- Input: `A_1_0001.jpg`
- Output: `1_A_0001.jpg`

---

## Note Generali

- Tutti gli script generano file CSV con il mapping `old_path → new_path`
- Gli script di augmentation salvano le nuove immagini nella stessa cartella delle originali
- Gli script di rename modificano direttamente i file originali
- Usare `--help` con qualsiasi script per vedere le opzioni disponibili
