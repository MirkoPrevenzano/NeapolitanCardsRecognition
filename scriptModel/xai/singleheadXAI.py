# Explainability (Captum) - Single-Head (40 classi)
# Script pensato per esecuzione headless (es. Kaggle): salva figure e crea ZIP finale.

import os
import gc
import zipfile
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torchvision import datasets, models, transforms
from captum.attr import IntegratedGradients, Occlusion, NoiseTunnel, visualization as viz

# ==========================================================
# 0) Config (da file) + Setup
# ==========================================================

def load_config(config_path):
    """Carica configurazione da file di testo in formato: key = value."""
    config = {}
    if not os.path.exists(config_path):
        return config
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.split('#')[0].strip()  # rimuove commenti inline
            # supporto opzionale a valori tra virgolette
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            config[key] = value
    return config


def parse_float_list(value):
    return [float(x.strip()) for x in value.split(',') if x.strip()]


def parse_int_tuple(value):
    return tuple(int(x.strip()) for x in value.split(',') if x.strip())


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'singleheadXAIConfig.txt')
cfg = load_config(CONFIG_PATH)
print(f"Configurazione caricata da: {CONFIG_PATH} | Chiavi: {len(cfg)}")

SEED = int(cfg.get('SEED', '42'))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo utilizzato: {device}")

# ==========================================================
# 1) Parametri (con default)
# ==========================================================

DATA_DIR = cfg.get('DATA_DIR', '/kaggle/input/datasets/davidedipierro24/versione3/datasetVersione3')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Assicurati di puntare al modello a 40 classi generato dal training
MODEL_PATH = cfg.get('MODEL_PATH', '/kaggle/input/models/davidedipierro24/40-classi/pytorch/default/1/best_model_40_classi.pth')
OUTPUT_DIR = cfg.get('OUTPUT_DIR', '/kaggle/working/captum_results_singlehead_40')
ZIP_PATH = cfg.get('ZIP_PATH', '/kaggle/working/captum_results_singlehead_40.zip')

NUM_CLASSES = int(cfg.get('NUM_CLASSES', '40'))

MEAN = parse_float_list(cfg.get('MEAN', '0.6055278, 0.57453782, 0.51603037'))
STD = parse_float_list(cfg.get('STD', '0.25722615, 0.27000131, 0.28853789'))

NT_TYPE = cfg.get('NT_TYPE', 'smoothgrad_sq')
NT_SAMPLES = int(cfg.get('NT_SAMPLES', '5'))
INTERNAL_BATCH_SIZE = int(cfg.get('INTERNAL_BATCH_SIZE', '5'))

OCC_SLIDING_WINDOW = parse_int_tuple(cfg.get('OCC_SLIDING_WINDOW', '3, 40, 40'))
OCC_STRIDES = parse_int_tuple(cfg.get('OCC_STRIDES', '3, 10, 10'))
BASELINE_OCCLUSION = parse_float_list(cfg.get('BASELINE_OCCLUSION', '0, 0, 0'))

FIGSIZE = parse_int_tuple(cfg.get('FIGSIZE', '16, 5'))
DPI = int(cfg.get('DPI', '150'))
ALPHA_OVERLAY_IG = float(cfg.get('ALPHA_OVERLAY_IG', '0.8'))
ALPHA_OVERLAY_OCC = float(cfg.get('ALPHA_OVERLAY_OCC', '0.7'))
OUTLIER_PERC = float(cfg.get('OUTLIER_PERC', '2'))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# 2) Dataset Single-Head (Standard ImageFolder)
# ==========================================================

# Converto i dati in tensori per una migliore gestione durante il calcolo delle attribuzioni.
# Applico standardizzazione usando la media e deviazione standard calcolate in precedenza (da calcoloMediaVarianza.py).
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
class_names = test_dataset.classes

print(f"Classi trovate nel test set: {len(class_names)}")
print(f"Immagini nel test set: {len(test_dataset)}")

# ==========================================================
# 3) Modello Single-Head (ResNet50)
# ==========================================================
def initialize_model(num_classes):
    # In XAI vogliamo replicare esattamente l'architettura usata nel checkpoint.
    # Qui istanzio ResNet50 senza pesi (weights=None) e carico poi lo state_dict.
    model = models.resnet50(weights=None)  # pesi custom dal tuo file
    num_ftrs = model.fc.in_features
    
    # Rimuoviamo nn.Sequential e nn.Dropout per far combaciare le chiavi del dizionario
    # (Il file .pth si aspetta 'fc.weight' e 'fc.bias')
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

# Rimuovi il parametro dropout_p che non ci serve più qui
model = initialize_model(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print("Modello single-head a 40 classi caricato correttamente.")

# ==========================================================
# 4) Wrapper per Captum (funzione forward)
# ==========================================================
def forward_func(inputs):
    return model(inputs)

# ==========================================================
# 5) Utility
# ==========================================================
def denormalize(tensor, mean, std):
    t = tensor.clone().detach().cpu()
    for c, m, s in zip(t, mean, std):
        c.mul_(s).add_(m)
    return torch.clamp(t, 0, 1)

def to_grayscale_3ch(img_rgb):
    gray = np.mean(img_rgb, axis=2)
    return np.stack((gray,) * 3, axis=-1)

def get_prediction(model, input_tensor):
    input_batch = input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_batch)
    probs = torch.softmax(outputs, dim=1)
    conf, pred = torch.max(probs, 1)
    return pred.item(), conf.item()

def safe_name(s):
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)

# ==========================================================
# 6) Explainability su TUTTO il test set
# ==========================================================
print("\nAnalisi di tutto il test set (single-head)...")
print("NOTA: Ottimizzazione VRAM attiva (Batch=1, Aggressive GC + empty_cache).")

ig = IntegratedGradients(forward_func)
nt = NoiseTunnel(ig)
occ = Occlusion(forward_func)

saved_files = []
all_indices = list(range(len(test_dataset)))

for i, idx in enumerate(all_indices):
    try:
        print(f"\n--- Immagine {i+1}/{len(all_indices)} (indice={idx}) ---")

        input_tensor, true_label_idx = test_dataset[idx]
        input_tensor_dev = input_tensor.to(device)

        pred_idx, conf = get_prediction(model, input_tensor_dev)

        true_name = class_names[true_label_idx]
        pred_name = class_names[pred_idx]

        print(f"  Vera: {true_name} | Pred: {pred_name} (Conf: {conf*100:.2f}%)")

        original_rgb = denormalize(input_tensor, MEAN, STD).permute(1, 2, 0).numpy()
        original_bw = to_grayscale_3ch(original_rgb)

        img_input = input_tensor_dev.unsqueeze(0)
        img_input.requires_grad = True

        # ----- IG SmoothGrad -----
        print("  Calcolo IG SmoothGrad...")
        # NoiseTunnel con nt_type=smoothgrad_sq: stabilizza la mappa IG riducendo rumore e rendendo più robuste le evidenze.
        attr_ig = nt.attribute(
            img_input, nt_type=NT_TYPE, nt_samples=NT_SAMPLES,
            target=pred_idx, internal_batch_size=INTERNAL_BATCH_SIZE
        )
        res_ig = np.transpose(attr_ig.squeeze(0).detach().cpu().numpy(), (1, 2, 0))
        del attr_ig
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ----- Occlusion -----
        print("  Calcolo Occlusion...")
        attr_occ = occ.attribute(
            img_input, strides=OCC_STRIDES,
            sliding_window_shapes=OCC_SLIDING_WINDOW,
            # Baseline a zeri nello spazio normalizzato (coerente con Normalize(MEAN, STD))
            baselines=torch.tensor(BASELINE_OCCLUSION).view(1, 3, 1, 1).to(img_input.device), 
            target=pred_idx
        )
        res_occ = np.transpose(attr_occ.squeeze(0).detach().cpu().numpy(), (1, 2, 0))
        del attr_occ, img_input
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ----- Plot: 1x3 layout -----
        fig, axs = plt.subplots(1, 3, figsize=FIGSIZE)
        fig.suptitle(
            f"Img {i} | Real: {true_name} | Pred: {pred_name}",
            fontsize=16
        )

        axs[0].imshow(original_rgb)
        axs[0].set_title("Originale")
        axs[0].axis('off')

        viz.visualize_image_attr(
            res_ig, original_bw, method="blended_heat_map", sign="positive",
            show_colorbar=True, title="IG SmoothGrad",
            plt_fig_axis=(fig, axs[1]),
            use_pyplot=False, alpha_overlay=ALPHA_OVERLAY_IG, outlier_perc=OUTLIER_PERC
        )

        viz.visualize_image_attr(
            res_occ, original_bw, method="blended_heat_map", sign="all",
            show_colorbar=True, title="Occlusion",
            plt_fig_axis=(fig, axs[2]),
            use_pyplot=False, alpha_overlay=ALPHA_OVERLAY_OCC, outlier_perc=OUTLIER_PERC
        )

        plt.tight_layout()

        filename = f"captum_{i:05d}_{safe_name(true_name)}_pred_{safe_name(pred_name)}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        fig.savefig(filepath, dpi=DPI, bbox_inches="tight", facecolor="white")

        plt.close('all')
        fig.clf()
        del fig, axs, res_ig, res_occ, original_rgb, original_bw
        gc.collect()

        saved_files.append(filepath)
        print(f"  Salvato: {filename}")

    except Exception as e:
        print(f"\nErrore su immagine {i}: {e}")
        torch.cuda.empty_cache()
        continue

# ==========================================================
# 7) ZIP finale
# ==========================================================
print("\nCreazione archivio ZIP...")
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for fpath in saved_files:
        zipf.write(fpath, os.path.basename(fpath))

print(f"Archivio creato: {ZIP_PATH}")
print(f"Contiene {len(saved_files)} immagini.")

print("\n" + "=" * 60)
print("ANALISI COMPLETATA (Single-Head 40 Classi)")
print(f"File ZIP: {ZIP_PATH}")
print(f"Cartella risultati: {OUTPUT_DIR}")
print("=" * 60)