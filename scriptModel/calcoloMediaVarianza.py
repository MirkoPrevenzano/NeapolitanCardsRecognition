# Calcolo di media e varianza sul dataset originale
import os
import cv2
import numpy as np
from tqdm import tqdm  # Per vedere la barra di avanzamento

# --- CONFIGURAZIONE ---
DATASET_PATH = '/kaggle/input/datasets/mirkoprevenzano/Versione3/datasetVersione3/train'  # Sostituisci con il tuo percorso
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

# Inizializzatori
channel_sum = np.zeros(3)       # Somma cumulativa dei canali (R, G, B)
channel_sum_sq = np.zeros(3)    # Somma cumulativa dei quadrati
pixel_count = 0                 # Numero totale di pixel

# Trova tutti i file immagine
image_paths = []
for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith(IMG_EXTENSIONS):
            image_paths.append(os.path.join(root, file))

print(f"Trovate {len(image_paths)} immagini.")

# --- CALCOLO ---
for path in tqdm(image_paths, desc="Elaborazione immagini"):
    # Leggi immagine
    img = cv2.imread(path)
    
    if img is None:
        continue

    # Converti da BGR (default OpenCV) a RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalizza i pixel tra 0 e 1 (opzionale, ma raccomandato per Deep Learning)
    img = img.astype('float32') / 255.0

    # Reshape dell'immagine in un array 2D (num_pixels, 3_canali)
    pixels = img.reshape(-1, 3)

    # Aggiorna le somme
    channel_sum += pixels.sum(axis=0)
    channel_sum_sq += (pixels ** 2).sum(axis=0)
    
    # Aggiorna il conteggio totale dei pixel
    pixel_count += pixels.shape[0]

# --- RISULTATI FINALI ---
# Media = Somma / N
mean = channel_sum / pixel_count

# Varianza = (Somma_Quadrati / N) - Media^2
variance = (channel_sum_sq / pixel_count) - (mean ** 2)

# Deviazione Standard = radice(Varianza)
std = np.sqrt(variance)

print("\n--- Risultati (ordine RGB) ---")
print(f"Media (Mean): {mean}")
print(f"Varianza (Var): {variance}")
print(f"Deviazione Standard (Std): {std}")