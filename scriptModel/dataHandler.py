# ==========================================================
# dataHandler.py
# Scopo: adattare il test set al modello multi-head e ricostruire
#        la confusion matrix completa 40x40 combinando le
#        predizioni delle due teste (numero + seme).
# NOTA: questo script va eseguito DOPO multiHeadModel.py,
#       perché usa variabili globali definite lì (device, model,
#       TEST_DIR, data_transforms, BATCH_SIZE, image_datasets, dataloaders).
# ==========================================================

import re                           # Espressioni regolari per estrarre numero e lettera
from sklearn.metrics import classification_report, confusion_matrix  # Metriche di valutazione
import seaborn as sns               # Grafici avanzati (heatmap)
import matplotlib.pyplot as plt     # Creazione di grafici
import numpy as np                  # Calcoli numerici
import torch                        # PyTorch
from torch.utils.data import DataLoader, Dataset  # DataLoader e classe base Dataset
from torchvision import datasets    # ImageFolder per caricare immagini da cartelle

# ==========================================
# 1. DATASET PERSONALIZZATO MULTI-LABEL
# ==========================================
# Versione del dataset che estrae DUE etichette dal nome della cartella:
# - label_num: valore della carta (0-9, dove 0="1" e 9="10")
# - label_char: seme della carta (0=A, 1=B, 2=C, 3=D)

class MultiHeadDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # Carica le immagini usando ImageFolder (etichette = nomi sottocartelle)
        self.dataset = datasets.ImageFolder(root_dir, transform=transform)
        self.classes = self.dataset.classes  # Lista nomi cartelle: ['1 A', '1 B', ...]
        
        # Mappa lettere → numeri
        self.letter_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        self.img_labels = []
        
        # Regex per estrarre numero e lettera dal nome cartella
        # Accetta formati: "1A", "1 A", "1  A", "10 D"
        regex_pattern = re.compile(r"(\d+)\s*([A-Z])")

        print(f"Caricamento dataset da: {root_dir}")
        print(f"Esempio classi trovate: {self.classes[:5]}")

        # Per ogni immagine, estrai le due etichette dal nome della cartella
        for _, class_idx in self.dataset.samples:
            class_name = self.classes[class_idx]  # Es: "10 A"
            match = regex_pattern.search(class_name)
            
            if match:
                num_str, char_str = match.groups()  # Es: ("10", "A")
                label_num = int(num_str) - 1         # "1"→0, "10"→9
                label_char = self.letter_map[char_str]  # "A"→0, "D"→3
                self.img_labels.append((label_num, label_char))
            else:
                # Se il nome non corrisponde al pattern, usa valori dummy
                self.img_labels.append((-1, -1)) 

    def __len__(self):
        """Restituisce il numero totale di immagini."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Restituisce (immagine, etichetta_numero, etichetta_seme).
        Il modello riceve SOLO pixel, MAI nomi file o metadati."""
        img, _ = self.dataset[idx]  # Carica immagine (ignora etichetta ImageFolder)
        label_num, label_char = self.img_labels[idx]
        return img, label_num, label_char

# ==========================================
# 2. RICREAZIONE DEL DATALOADER DI TEST
# ==========================================
# Sostituiamo il dataset e il dataloader del test set con la versione
# MultiHeadDataset, che restituisce due etichette invece di una
# NOTA: TEST_DIR, data_transforms, BATCH_SIZE sono variabili globali
#       definite in multiHeadModel.py
image_datasets['test'] = MultiHeadDataset(TEST_DIR, data_transforms['test'])
dataloaders['test'] = DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ==========================================
# 3. FUNZIONE DI VALUTAZIONE MULTI-HEAD
# ==========================================

def evaluate_multi_head(model, dataloader):
    """Valuta il modello multi-head e restituisce le predizioni
    separate per numeri e lettere."""
    model.eval()  # Modalità valutazione (disattiva dropout)
    
    # Liste per raccogliere risultati
    all_preds_num = []      # Predizioni dei numeri
    all_labels_num = []     # Etichette vere dei numeri
    all_preds_char = []     # Predizioni delle lettere
    all_labels_char = []    # Etichette vere delle lettere
    
    # Disabilita il calcolo dei gradienti (non servono in valutazione)
    with torch.no_grad():
        for inputs, labels_num, labels_char in dataloader:
            inputs = inputs.to(device)          # Sposta immagini su GPU
            labels_num = labels_num.to(device)  # Sposta etichette numeri su GPU
            labels_char = labels_char.to(device) # Sposta etichette lettere su GPU
            
            # Forward: ottieni le predizioni delle due teste
            out_num, out_char = model(inputs)
            
            # Prendi la classe con probabilità massima per ciascuna testa
            _, preds_num = torch.max(out_num, 1)
            _, preds_char = torch.max(out_char, 1)
            
            # Salva risultati su CPU
            all_preds_num.extend(preds_num.cpu().numpy())
            all_labels_num.extend(labels_num.cpu().numpy())
            all_preds_char.extend(preds_char.cpu().numpy())
            all_labels_char.extend(labels_char.cpu().numpy())
    
    # Restituisce due tuple: (vere_num, pred_num), (vere_char, pred_char)
    return (np.array(all_labels_num), np.array(all_preds_num)), \
           (np.array(all_labels_char), np.array(all_preds_char))

print("Ricalcolo predizioni sul Test Set...")
(y_true_num, y_pred_num), (y_true_char, y_pred_char) = evaluate_multi_head(model, dataloaders['test'])

# ==========================================
# 4. RICOSTRUZIONE DELLA MATRICE 40x40
# ==========================================
# Le due teste predicono numero (1-10) e lettera (A-D) separatamente.
# Per ottenere la confusion matrix delle 40 carte originali,
# ricombiniamo le predizioni: (num, char) → nome classe originale
print("Generazione Matrice di Confusione...")

# Lista delle classi originali dal dataset test (es. ['1 A', '1 B', ..., '10 D'])
original_classes = image_datasets['test'].classes
regex_pattern = re.compile(r"(\d+)\s*([A-Z])")
letter_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

# Crea una mappa: (indice_num, indice_char) → nome_classe_originale
# Es: (9, 0) → "10 A" (perché indice 9 = valore 10, indice 0 = seme A)
pair_to_original_name = {}

for cls_name in original_classes:
    match = regex_pattern.search(cls_name)
    if match:
        n_str, c_str = match.groups()
        n_idx = int(n_str) - 1          # "10" → indice 9
        c_idx = letter_map[c_str]       # "A" → indice 0
        pair_to_original_name[(n_idx, c_idx)] = cls_name

# Mappa: nome classe → indice globale (0-39)
class_to_idx = {cls_name: i for i, cls_name in enumerate(original_classes)}

y_pred_combined = []  # Predizioni come indici 0-39
y_true_combined = []  # Verità come indici 0-39

# Per ogni immagine, ricombina le predizioni delle due teste
for i in range(len(y_pred_num)):
    # --- Coppia Predetta: (num_predetto, char_predetto) → nome classe ---
    pred_pair = (y_pred_num[i], y_pred_char[i])
    if pred_pair in pair_to_original_name:
        pred_name = pair_to_original_name[pred_pair]  # Es: "10 A"
        y_pred_combined.append(class_to_idx[pred_name])  # Es: indice 0
    else:
        y_pred_combined.append(-1)  # Combinazione impossibile

    # --- Coppia Vera: (num_vero, char_vero) → nome classe ---
    true_pair = (y_true_num[i], y_true_char[i])
    if true_pair in pair_to_original_name:
        true_name = pair_to_original_name[true_pair]
        y_true_combined.append(class_to_idx[true_name])
    else:
        y_true_combined.append(-1)

# Rimuovi eventuali predizioni impossibili (valore -1)
valid_mask = [i for i, x in enumerate(y_pred_combined) if x != -1 and y_true_combined[i] != -1]
y_pred_clean = [y_pred_combined[i] for i in valid_mask]
y_true_clean = [y_true_combined[i] for i in valid_mask]

# ==========================================
# 5. VISUALIZZAZIONE DELLA MATRICE 40x40
# ==========================================

if len(y_true_clean) > 0:
    # Crea la confusion matrix 40x40
    plt.figure(figsize=(24, 20))
    cm_full = confusion_matrix(y_true_clean, y_pred_clean)
    # Heatmap: ogni cella mostra quante immagini della classe Y sono state predette come classe X
    sns.heatmap(cm_full, annot=True, fmt='d', cmap='Purples',
                xticklabels=original_classes,   # Nomi classi sull'asse X
                yticklabels=original_classes)    # Nomi classi sull'asse Y
    plt.title('Confusion Matrix Completa (40 Classi)')
    plt.ylabel('Vero')      # Asse Y: classe vera
    plt.xlabel('Predetto')  # Asse X: classe predetta
    plt.xticks(rotation=90) # Ruota le etichette per leggibilità
    plt.show()
    
    # Report testuale: precision, recall, F1-score per ciascuna delle 40 carte
    print(classification_report(y_true_clean, y_pred_clean, target_names=original_classes, labels=range(len(original_classes))))
else:
    # Se nessuna classe corrisponde, c'è un errore nei nomi delle cartelle
    print("Errore: Nessuna classe corrisponde. Controlla che i nomi cartelle contengano Numeri e Lettere.")