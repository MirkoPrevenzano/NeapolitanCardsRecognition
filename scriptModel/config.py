# ==========================================================
# config.py
# Scopo: calcolare la media (mean) e la deviazione standard
#        (std) dei pixel per canale RGB sulle immagini del
#        SOLO training set. Questi valori servono poi per
#        normalizzare le immagini durante il training e il test.
# ==========================================================

import os                       # Per navigare tra cartelle e file del sistema operativo

import cv2                      # OpenCV: libreria per leggere e manipolare immagini
import numpy as np              # NumPy: libreria per calcoli numerici su array
from tqdm import tqdm            # tqdm: mostra una barra di avanzamento durante i cicli

# --- CONFIGURAZIONE ---
# Percorso della cartella contenente le immagini di SOLO training
# (IMPORTANTE: calcoliamo media/std solo su train per evitare data leakage)
DATASET_PATH = '/kaggle/input/original/dataset_original/train'
# Estensioni dei file immagine da cercare
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

# --- Variabili accumulatrici ---
# Servono per calcolare media e std in modo incrementale (senza caricare tutto in RAM)
channel_sum = np.zeros(3)       # Somma cumulativa dei valori dei pixel per ogni canale (R, G, B)
channel_sum_sq = np.zeros(3)    # Somma cumulativa dei quadrati dei pixel (serve per la varianza)
pixel_count = 0                 # Contatore totale di tutti i pixel elaborati

# --- Raccolta di tutti i percorsi delle immagini ---
# os.walk scorre ricorsivamente tutte le sottocartelle
image_paths = []
for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        # Controlla se il file ha un'estensione immagine valida
        if file.lower().endswith(IMG_EXTENSIONS):
            image_paths.append(os.path.join(root, file))

print(f"Trovate {len(image_paths)} immagini.")

# --- CALCOLO INCREMENTALE DI MEDIA E STD ---
# Per ogni immagine: leggi i pixel, sommali, e aggiorna i contatori
for path in tqdm(image_paths, desc="Elaborazione immagini"):
    # Leggi l'immagine dal disco (OpenCV la carica come array di pixel)
    img = cv2.imread(path)
    
    # Se l'immagine è corrotta o illeggibile, saltala
    if img is None:
        continue

    # OpenCV carica i colori in ordine BGR; li convertiamo in RGB (ordine standard)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Porta i valori dei pixel da [0, 255] a [0.0, 1.0] (necessario per il deep learning)
    img = img.astype('float32') / 255.0

    # Trasforma l'immagine da matrice 3D (altezza x larghezza x 3) a tabella 2D (N_pixel x 3)
    pixels = img.reshape(-1, 3)

    # Accumula la somma di tutti i pixel per canale
    channel_sum += pixels.sum(axis=0)
    # Accumula la somma dei quadrati (serve per calcolare la varianza)
    channel_sum_sq += (pixels ** 2).sum(axis=0)
    
    # Aggiorna il numero totale di pixel elaborati
    pixel_count += pixels.shape[0]

# --- RISULTATI FINALI ---
# Formula della media: mean = somma_pixel / numero_pixel
mean = channel_sum / pixel_count

# Formula della varianza: var = E[X^2] - (E[X])^2
# dove E[X^2] = somma_quadrati / N, e E[X] = mean
variance = (channel_sum_sq / pixel_count) - (mean ** 2)

# La deviazione standard è la radice quadrata della varianza
std = np.sqrt(variance)

# Stampa i risultati: questi 3 valori (uno per R, G, B) verranno poi
# usati in transforms.Normalize(MEAN, STD) nei file di training
print("\n--- Risultati (ordine RGB) ---")
print(f"Media (Mean): {mean}")
print(f"Varianza (Var): {variance}")
print(f"Deviazione Standard (Std): {std}")