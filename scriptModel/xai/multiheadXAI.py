# Explainability (Captum) - Multi-Head (Numeri + Lettere)
# Script pensato per esecuzione headless (es. Kaggle): salva figure e crea ZIP finale.

import os
import re
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
from torch.utils.data import Dataset
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


def parse_bool(value):
    return str(value).lower() in ('true', '1', 'yes', 'si')


def parse_float_list(value):
    return [float(x.strip()) for x in value.split(',') if x.strip()]


def parse_int_tuple(value):
    return tuple(int(x.strip()) for x in value.split(',') if x.strip())


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'multiheadXAIConfig.txt')
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
MODEL_PATH = cfg.get('MODEL_PATH', '/kaggle/input/models/davidedipierro24/14-classi/pytorch/default/1/best_model.pth')
OUTPUT_DIR = cfg.get('OUTPUT_DIR', '/kaggle/working/captum_results_multihead_opt')
ZIP_PATH = cfg.get('ZIP_PATH', '/kaggle/working/captum_results_multihead_opt.zip')

NUM_CLASSES_NUMBER = int(cfg.get('NUM_CLASSES_NUMBER', '10'))
NUM_CLASSES_LETTER = int(cfg.get('NUM_CLASSES_LETTER', '4'))
DROPOUT_P = float(cfg.get('DROPOUT_P', '0.3'))

MEAN = parse_float_list(cfg.get('MEAN', '0.6055278, 0.57453782, 0.51603037'))
STD = parse_float_list(cfg.get('STD', '0.25722615, 0.27000131, 0.28853789'))

NT_TYPE = cfg.get('NT_TYPE', 'smoothgrad_sq')
NT_SAMPLES = int(cfg.get('NT_SAMPLES', '5'))
INTERNAL_BATCH_SIZE_NUM = int(cfg.get('INTERNAL_BATCH_SIZE_NUM', '5'))
INTERNAL_BATCH_SIZE_CHAR = int(cfg.get('INTERNAL_BATCH_SIZE_CHAR', '1'))

OCC_SLIDING_WINDOW = parse_int_tuple(cfg.get('OCC_SLIDING_WINDOW', '3, 40, 40'))
OCC_STRIDES = parse_int_tuple(cfg.get('OCC_STRIDES', '3, 10, 10'))
BASELINE_OCCLUSION = parse_float_list(cfg.get('BASELINE_OCCLUSION', '0, 0, 0'))

FIGSIZE = parse_int_tuple(cfg.get('FIGSIZE', '16, 10'))
DPI = int(cfg.get('DPI', '150'))
ALPHA_OVERLAY_IG = float(cfg.get('ALPHA_OVERLAY_IG', '0.8'))
ALPHA_OVERLAY_OCC = float(cfg.get('ALPHA_OVERLAY_OCC', '0.7'))
OUTLIER_PERC = float(cfg.get('OUTLIER_PERC', '2'))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# 2) Dataset Multi-Head
# ==========================================================
class MultiHeadDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # Carico dataset usando ImageFolder per sfruttare la struttura delle cartelle
        self.dataset = datasets.ImageFolder(root_dir, transform=transform)
        self.classes = self.dataset.classes
        self.letter_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        self.inv_letter_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

        self.img_labels = []
        for _, class_idx in self.dataset.samples:
            class_name = self.classes[class_idx]
            # Regex per estrarre numero e lettera, gestendo possibili separatori (es. "10A", "10_A", "10-A")
            match = re.search(r"(\d+)\s*[_\-\s]*([A-Z])", class_name)
            if match:
                num_str, char_str = match.groups()
                label_num = int(num_str) - 1
                label_char = self.letter_map[char_str]
                self.img_labels.append((label_num, label_char))
            else:
                print(f"Attenzione: Impossibile parsare la cartella: '{class_name}'")
                self.img_labels.append((0, 0))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        label_num, label_char = self.img_labels[idx]
        return img, label_num, label_char

# Converto i dati in tensori per una migliore gestione durante il calcolo delle attribuzioni.
# Applico standardizzazione usando la media e deviazione standard calcolate in precedenza (da calcoloMediaVarianza.py).
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

test_dataset = MultiHeadDataset(TEST_DIR, test_transform)
folder_classes = test_dataset.classes

names_num = [str(i) for i in range(1, NUM_CLASSES_NUMBER + 1)]
names_char = ['A', 'B', 'C', 'D']  # mapping coerente con letter_map

print(f"Classi cartelle trovate: {len(folder_classes)}")
print(f"Immagini nel test set: {len(test_dataset)}")

# ==========================================================
# 3) Modello Multi-Head
# ==========================================================
class MultiHeadResNet50(nn.Module):
    def __init__(self, num_classes_num, num_classes_char, dropout_p=0.3):
        super(MultiHeadResNet50, self).__init__()
        # In XAI vogliamo replicare esattamente l'architettura usata nel checkpoint.
        # Qui istanzio ResNet50 senza pesi (weights=None) e carico poi lo state_dict.
        self.base_model = models.resnet50(weights=None)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.dropout = nn.Dropout(p=dropout_p)
        self.head_num = nn.Linear(num_ftrs, num_classes_num)
        self.head_char = nn.Linear(num_ftrs, num_classes_char)

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        out_num = self.head_num(x)
        out_char = self.head_char(x)
        return out_num, out_char

model = MultiHeadResNet50(NUM_CLASSES_NUMBER, NUM_CLASSES_LETTER, dropout_p=DROPOUT_P)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print("Modello multi-head caricato correttamente.")

# ==========================================================
# 4) Wrapper per Captum (funzioni forward)
# ==========================================================
def forward_func_num(inputs):
    out_num, _ = model(inputs)
    return out_num

def forward_func_char(inputs):
    _, out_char = model(inputs)
    return out_char

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
    return np.stack((gray,)*3, axis=-1)

def get_prediction_multihead(model, input_tensor):
    input_batch = input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        out_num, out_char = model(input_batch)
    probs_num = torch.softmax(out_num, dim=1)
    probs_char = torch.softmax(out_char, dim=1)
    conf_num, pred_num = torch.max(probs_num, 1)
    conf_char, pred_char = torch.max(probs_char, 1)
    return (pred_num.item(), conf_num.item()), (pred_char.item(), conf_char.item())

def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)

def format_class_name(num_idx, char_idx):
    return f"{num_idx + 1}{names_char[char_idx]}"

# ==========================================================
# 6) Explainability su TUTTO il test set
# ==========================================================
print("\nAnalisi di tutto il test set (multi-head)...")
print("NOTA: Ottimizzazione VRAM attiva (Batch=1, Aggressive GC + empty_cache).")

ig_num = IntegratedGradients(forward_func_num)
ig_char = IntegratedGradients(forward_func_char)
nt_num = NoiseTunnel(ig_num)
nt_char = NoiseTunnel(ig_char)
occ_num = Occlusion(forward_func_num)
occ_char = Occlusion(forward_func_char)

saved_files = []
all_indices = list(range(len(test_dataset)))

for i, idx in enumerate(all_indices):
    try:
        print(f"\n--- Immagine {i+1}/{len(all_indices)} (indice={idx}) ---")

        input_tensor, true_num, true_char = test_dataset[idx]
        input_tensor_dev = input_tensor.to(device)

        (pred_num, conf_num), (pred_char, conf_char) = get_prediction_multihead(model, input_tensor_dev)

        true_name = format_class_name(true_num, true_char)
        pred_name = format_class_name(pred_num, pred_char)

        real_lbl_num = true_num + 1
        real_lbl_char = names_char[true_char]
        pred_lbl_num = pred_num + 1
        pred_lbl_char = names_char[pred_char]

        print(f"  Vera: {true_name} | Pred: {pred_name}")

        original_rgb = denormalize(input_tensor, MEAN, STD).permute(1, 2, 0).numpy()
        original_bw = to_grayscale_3ch(original_rgb)

        img_input = input_tensor_dev.unsqueeze(0)
        img_input.requires_grad = True

        # ----- IG SmoothGrad: HEAD NUMERI -----
        print("  IG SmoothGrad head numeri...")
        # NoiseTunnel con nt_type=smoothgrad_sq: stabilizza la mappa IG riducendo rumore e rendendo più robuste le evidenze.
        attr_ig_num = nt_num.attribute(
            img_input, nt_type=NT_TYPE, nt_samples=NT_SAMPLES,
            target=pred_num, internal_batch_size=INTERNAL_BATCH_SIZE_NUM
        )
        res_ig_num = np.transpose(attr_ig_num.squeeze(0).detach().cpu().numpy(), (1, 2, 0))
        del attr_ig_num
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ----- IG SmoothGrad: HEAD LETTERE -----
        print("  IG SmoothGrad head lettere...")
        attr_ig_char = nt_char.attribute(
            img_input, nt_type=NT_TYPE, nt_samples=NT_SAMPLES,
            target=pred_char, internal_batch_size=INTERNAL_BATCH_SIZE_CHAR
        )
        res_ig_char = np.transpose(attr_ig_char.squeeze(0).detach().cpu().numpy(), (1, 2, 0))
        del attr_ig_char
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        del img_input

        # ----- Occlusion: HEAD NUMERI -----
        print("  Occlusion head numeri...")
        img_occ = input_tensor_dev.unsqueeze(0)
        attr_occ_num = occ_num.attribute(
            img_occ, strides=OCC_STRIDES,
            sliding_window_shapes=OCC_SLIDING_WINDOW,
            # Baseline a zeri nello spazio normalizzato (coerente con Normalize(MEAN, STD))
            baselines=torch.tensor(BASELINE_OCCLUSION).view(1, 3, 1, 1).to(img_occ.device), target=pred_num
        )
        res_occ_num = np.transpose(attr_occ_num.squeeze(0).detach().cpu().numpy(), (1, 2, 0))
        del attr_occ_num
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ----- Occlusion: HEAD LETTERE -----
        print("  Occlusion head lettere...")
        attr_occ_char = occ_char.attribute(
            img_occ, strides=OCC_STRIDES,
            sliding_window_shapes=OCC_SLIDING_WINDOW,
            baselines=torch.tensor(BASELINE_OCCLUSION).view(1, 3, 1, 1).to(img_occ.device), target=pred_char
        )
        res_occ_char = np.transpose(attr_occ_char.squeeze(0).detach().cpu().numpy(), (1, 2, 0))
        del attr_occ_char, img_occ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ----- Plot: 2x3 layout (come prima) -----
        fig, axs = plt.subplots(2, 3, figsize=FIGSIZE)
        fig.suptitle(
            f"Img {i} | Real: {real_lbl_num}-{real_lbl_char} | Pred: {pred_lbl_num}-{pred_lbl_char}",
            fontsize=16
        )

        # Riga 1 (Numeri)
        axs[0, 0].imshow(original_rgb)
        axs[0, 0].set_title("Orig")
        axs[0, 0].axis('off')

        viz.visualize_image_attr(
            res_ig_num, original_bw, method="blended_heat_map", sign="positive",
            show_colorbar=True, title="IG Smooth Num",
            plt_fig_axis=(fig, axs[0, 1]),
            use_pyplot=False, alpha_overlay=ALPHA_OVERLAY_IG, outlier_perc=OUTLIER_PERC
        )

        viz.visualize_image_attr(
            res_occ_num, original_bw, method="blended_heat_map", sign="all",
            show_colorbar=True, title="Occlusion Num",
            plt_fig_axis=(fig, axs[0, 2]),
            use_pyplot=False, alpha_overlay=ALPHA_OVERLAY_OCC, outlier_perc=OUTLIER_PERC
        )

        # Riga 2 (Lettere)
        axs[1, 0].imshow(original_rgb)
        axs[1, 0].set_title("Orig")
        axs[1, 0].axis('off')

        viz.visualize_image_attr(
            res_ig_char, original_bw, method="blended_heat_map", sign="positive",
            show_colorbar=True, title="IG Smooth Char",
            plt_fig_axis=(fig, axs[1, 1]),
            use_pyplot=False, alpha_overlay=ALPHA_OVERLAY_IG, outlier_perc=OUTLIER_PERC
        )

        viz.visualize_image_attr(
            res_occ_char, original_bw, method="blended_heat_map", sign="all",
            show_colorbar=True, title="Occlusion Char",
            plt_fig_axis=(fig, axs[1, 2]),
            use_pyplot=False, alpha_overlay=ALPHA_OVERLAY_OCC, outlier_perc=OUTLIER_PERC
        )

        plt.tight_layout()

        filename = f"captum_{i:05d}_{safe_name(true_name)}_pred_{safe_name(pred_name)}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        fig.savefig(filepath, dpi=DPI, bbox_inches="tight", facecolor="white")

        plt.close('all')
        fig.clf()
        del fig, axs, res_ig_num, res_ig_char, res_occ_num, res_occ_char, original_rgb, original_bw
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
print("ANALISI COMPLETATA (Multi-Head)")
print(f"File ZIP: {ZIP_PATH}")
print(f"Cartella risultati: {OUTPUT_DIR}")
print("=" * 60)