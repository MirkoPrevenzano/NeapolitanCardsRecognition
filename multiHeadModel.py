# ==========================================================
# multiHeadModel.py
# Scopo: addestrare un modello ResNet50 con DUE teste di output:
#   - Testa 1 (num): predice il VALORE della carta (1-10)
#   - Testa 2 (char): predice il SEME della carta (A, B, C, D)
# Questo approccio scompone il problema 40-classi in due
# sotto-problemi più semplici (10 classi + 4 classi).
# ==========================================================

import os                           # Per gestire percorsi di file e cartelle
import torch                        # PyTorch: framework principale per reti neurali
import torch.nn as nn               # Moduli per costruire strati della rete (Linear, ecc.)
import torch.optim as optim         # Ottimizzatori (Adam) per aggiornare i pesi
from torch.utils.data import DataLoader, Dataset  # DataLoader per batch, Dataset per dati personalizzati
from torchvision import datasets, models, transforms  # Modelli pre-addestrati e trasformazioni immagini
import numpy as np                  # Calcoli numerici su array
import matplotlib.pyplot as plt     # Creazione di grafici
import seaborn as sns               # Grafici avanzati (heatmap)
from sklearn.metrics import confusion_matrix, classification_report  # Metriche di valutazione
import time                         # Per misurare il tempo di training
import copy                         # Per copiare i pesi del miglior modello
import re                           # Espressioni regolari per estrarre numero e lettera dai nomi cartelle

# ==========================================
# 1. CONFIGURAZIONE E IPERPARAMETRI
# ==========================================

# Sceglie automaticamente se usare la GPU (più veloce) o la CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo utilizzato: {device}")

# Cartella principale del dataset
DATA_DIR = '/kaggle/input/augmented/augmentationDataset'

# --- Iperparametri ---
BATCH_SIZE = 32          # Quante immagini vengono elaborate insieme ad ogni passo
EPOCHS = 30              # Quante volte il modello vede l'intero dataset
LEARNING_RATE = 0.001    # Velocità di apprendimento

# Dimensioni delle due teste di output:
NUM_CLASSES_NUMBER = 10  # Valori delle carte: 1, 2, 3, ..., 10
NUM_CLASSES_LETTER = 4   # Semi delle carte: A (coppe), B (denari), C (spade), D (bastoni)

# Media e std calcolate SOLO sul training set (vedi config.py)
MEAN = [0.64737422, 0.63253646, 0.59358844]
STD = [0.1556388, 0.17054404, 0.19233133]

# ==========================================
# 2. DATASET PERSONALIZZATO MULTI-LABEL
# ==========================================
# Questa classe personalizzata estrae DUE etichette separate da ogni immagine:
# - label_num: il valore numerico della carta (0-9, dove 0="1" e 9="10")
# - label_char: il seme della carta (0=A, 1=B, 2=C, 3=D)
# Le etichette vengono estratte dal NOME DELLA CARTELLA (es. "10 A" → num=9, char=0)

class MultiHeadDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # Carica le immagini dalle cartelle usando ImageFolder
        # Le etichette vengono assegnate in base al nome della sottocartella
        self.dataset = datasets.ImageFolder(root_dir, transform=transform)
        self.classes = self.dataset.classes  # Lista nomi cartelle: ['1 A', '1 B', ..., '10 D']
        
        # Mappa per convertire lettere in numeri: A→0, B→1, C→2, D→3
        self.letter_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        
        # Per ogni immagine, estrai le due etichette dal nome della cartella
        self.img_labels = []
        for _, class_idx in self.dataset.samples:
            class_name = self.classes[class_idx]  # Es: "10 A"
            
            # Regex: cerca un numero seguito da spazi e una lettera maiuscola
            # Es: "10 A" → groups: ("10", "A")
            match = re.match(r"(\d+)\s+([A-Z])", class_name)
            
            if match:
                num_str, char_str = match.groups()  # Estrai numero e lettera
                label_num = int(num_str) - 1         # Converti: "1"→0, "10"→9
                label_char = self.letter_map[char_str]  # Converti: "A"→0, "D"→3
                self.img_labels.append((label_num, label_char))
            else:
                print(f"Attenzione: Impossibile parsare la cartella: '{class_name}'")
                self.img_labels.append((0, 0))  # Valore di fallback

    def __len__(self):
        """Restituisce il numero totale di immagini nel dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Restituisce una singola immagine con le sue due etichette.
        Il modello riceve SOLO il tensore dell'immagine e le etichette numeriche,
        MAI il nome del file o altri metadati."""
        img, _ = self.dataset[idx]  # Carica l'immagine (ignora l'etichetta originale di ImageFolder)
        label_num, label_char = self.img_labels[idx]  # Recupera le due etichette
        return img, label_num, label_char

# --- Trasformazioni da applicare alle immagini ---
# ToTensor(): converte l'immagine in tensore PyTorch (valori 0.0-1.0)
# Normalize(): normalizza ogni canale RGB con media e std calcolate solo su training
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]),
}

# Percorsi delle tre cartelle di split
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid') 
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Crea i dataset usando la classe personalizzata MultiHeadDataset
# che restituisce (immagine, etichetta_numero, etichetta_seme)
image_datasets = {
    'train': MultiHeadDataset(TRAIN_DIR, data_transforms['train']),
    'valid': MultiHeadDataset(VALID_DIR, data_transforms['valid']),
    'test': MultiHeadDataset(TEST_DIR, data_transforms['test'])
}

# DataLoader: raggruppa le immagini in batch per il training
# shuffle=True per train/valid: mescola i dati ad ogni epoca
# shuffle=False per test: mantiene l'ordine per la valutazione finale
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
    'valid': DataLoader(image_datasets['valid'], batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
    'test': DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
}

# Conta quante immagini ci sono in ogni split
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
print(f"Dataset sizes: {dataset_sizes}")

# ==========================================
# 3. MODELLO MULTI-HEAD (ResNet50 Modificata)
# ==========================================
# Architettura:
#   ResNet50 (backbone) → estrae 2048 feature dall'immagine
#       ├── Testa Numeri (Linear 2048→10): predice il valore (1-10)
#       └── Testa Lettere (Linear 2048→4):  predice il seme (A-D)

class MultiHeadResNet50(nn.Module):
    def __init__(self, num_classes_num, num_classes_char):
        super(MultiHeadResNet50, self).__init__()
        
        # Carica ResNet50 pre-addestrata su ImageNet (conosce già forme e texture)
        self.base_model = models.resnet50(weights='IMAGENET1K_V1')
        
        # Numero di feature estratte dalla rete base (2048 per ResNet50)
        num_ftrs = self.base_model.fc.in_features
        
        # Sostituiamo l'ultimo strato con Identity ("non fare nulla")
        # così otteniamo direttamente le 2048 feature grezze
        self.base_model.fc = nn.Identity()
        
        # Creiamo le due teste separate:
        self.head_num = nn.Linear(num_ftrs, num_classes_num)   # 2048 → 10 (valori)
        self.head_char = nn.Linear(num_ftrs, num_classes_char)  # 2048 → 4 (semi)
        
    def forward(self, x):
        """Passaggio in avanti: l'immagine attraversa la rete base,
        poi le feature vengono inviate alle due teste separate."""
        x = self.base_model(x)      # Estrai le 2048 feature dall'immagine
        out_num = self.head_num(x)   # Predici il valore (1-10)
        out_char = self.head_char(x) # Predici il seme (A-D)
        return out_num, out_char

# Crea il modello e spostalo sulla GPU
model = MultiHeadResNet50(NUM_CLASSES_NUMBER, NUM_CLASSES_LETTER)
model = model.to(device)

# CrossEntropyLoss: funzione di errore per classificazione
# Viene usata separatamente per ciascuna testa
criterion = nn.CrossEntropyLoss()

# Adam: ottimizzatore che aggiorna TUTTI i parametri del modello
# (backbone + entrambe le teste) → fine-tuning completo
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 

# ==========================================
# 4. LOOP DI TRAINING MULTI-HEAD
# ==========================================

def train_model(model, criterion, optimizer, num_epochs=10):
    """Addestra il modello multi-head per il numero di epoche specificato.
    Ad ogni epoca: fase di training (impara) + fase di validazione (valuta)."""
    since = time.time()  # Tempo di inizio
    
    # Salva i pesi del miglior modello (basato sulla media delle due accuratezze)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc_avg = 0.0  # Migliore accuratezza media (num + char) / 2
    
    # Dizionario per salvare lo storico per i grafici
    history = {
        'train_loss': [], 'valid_loss': [],
        'train_acc_num': [], 'valid_acc_num': [],     # Accuratezza sui numeri
        'train_acc_char': [], 'valid_acc_char': []    # Accuratezza sulle lettere
    }

    # Ciclo principale: ripeti per num_epochs volte
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Ogni epoca ha due fasi: training e validazione
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()   # Modalità training: attiva dropout e batch norm
            else:
                model.eval()    # Modalità valutazione: disattiva dropout

            running_loss = 0.0          # Errore accumulato
            running_corrects_num = 0    # Predizioni corrette per i numeri
            running_corrects_char = 0   # Predizioni corrette per le lettere

            # Scorri tutti i batch (ogni batch contiene BATCH_SIZE immagini)
            for inputs, labels_num, labels_char in dataloaders[phase]:
                inputs = inputs.to(device)          # Sposta immagini su GPU
                labels_num = labels_num.to(device)  # Sposta etichette numeri su GPU
                labels_char = labels_char.to(device) # Sposta etichette lettere su GPU

                optimizer.zero_grad()  # Azzera i gradienti dal passo precedente

                # Calcola i gradienti solo in training, non in validazione
                with torch.set_grad_enabled(phase == 'train'):
                    # Forward: passa le immagini nella rete e ottieni DUE output
                    outputs_num, outputs_char = model(inputs)
                    
                    # Prendi la classe con probabilità massima per ciascuna testa
                    _, preds_num = torch.max(outputs_num, 1)    # Predizione valore
                    _, preds_char = torch.max(outputs_char, 1)  # Predizione seme
                    
                    # Calcola l'errore separatamente per ogni testa
                    loss_num = criterion(outputs_num, labels_num)    # Errore sui numeri
                    loss_char = criterion(outputs_char, labels_char) # Errore sulle lettere
                    
                    # Loss totale = somma dei due errori
                    loss = loss_num + loss_char

                    # Backward + aggiornamento pesi (solo durante il training)
                    if phase == 'train':
                        loss.backward()      # Calcola i gradienti
                        optimizer.step()     # Aggiorna i pesi

                # Accumula errore e predizioni corrette
                running_loss += loss.item() * inputs.size(0)
                running_corrects_num += torch.sum(preds_num == labels_num.data)
                running_corrects_char += torch.sum(preds_char == labels_char.data)

            # Calcola loss e accuratezza medie per questa epoca
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc_num = running_corrects_num.double() / dataset_sizes[phase]    # Acc numeri
            epoch_acc_char = running_corrects_char.double() / dataset_sizes[phase]  # Acc lettere
            
            # Media delle due accuratezze: usata per scegliere il miglior modello
            epoch_acc_avg = (epoch_acc_num + epoch_acc_char) / 2

            print(f'{phase} Loss: {epoch_loss:.4f} | Acc Num: {epoch_acc_num:.4f} | Acc Char: {epoch_acc_char:.4f}')

            # Salva nello storico per generare i grafici dopo
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc_num'].append(epoch_acc_num.item())
                history['train_acc_char'].append(epoch_acc_char.item())
            else:
                history['valid_loss'].append(epoch_loss)
                history['valid_acc_num'].append(epoch_acc_num.item())
                history['valid_acc_char'].append(epoch_acc_char.item())

            # Se l'accuratezza media di validazione è la migliore, salva i pesi
            if phase == 'valid' and epoch_acc_avg > best_acc_avg:
                best_acc_avg = epoch_acc_avg
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    # Stampa durata totale del training
    time_elapsed = time.time() - since
    print(f'Training completato in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Miglior Avg Val Acc: {best_acc_avg:.4f}')

    # Ricarica i pesi del miglior modello
    model.load_state_dict(best_model_wts)
    return model, history

# Avvia il training del modello
model, history = train_model(model, criterion, optimizer, num_epochs=EPOCHS)

# ==========================================
# 5. VISUALIZZAZIONE TRAINING
# ==========================================

# Crea una figura con 3 grafici affiancati
plt.figure(figsize=(18, 5))

# Grafico 1: andamento dell'errore totale (loss_num + loss_char)
plt.subplot(1, 3, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['valid_loss'], label='Valid Loss')
plt.title('Loss Totale')
plt.legend()

# Grafico 2: accuratezza della testa che predice i numeri (1-10)
plt.subplot(1, 3, 2)
plt.plot(history['train_acc_num'], label='Train Num Acc')
plt.plot(history['valid_acc_num'], label='Valid Num Acc')
plt.title('Accuratezza Numeri (1-10)')
plt.legend()

# Grafico 3: accuratezza della testa che predice le lettere (A-D)
plt.subplot(1, 3, 3)
plt.plot(history['train_acc_char'], label='Train Char Acc')
plt.plot(history['valid_acc_char'], label='Valid Char Acc')
plt.title('Accuratezza Lettere (A-D)')
plt.legend()

plt.show()  # Mostra i grafici

# ==========================================
# 6. VALUTAZIONE SUL TEST SET & METRICHE
# ==========================================

def evaluate_multi_head(model, dataloader):
    """Valuta il modello multi-head sul test set.
    Restituisce le etichette vere e predette per entrambe le teste."""
    model.eval()  # Modalità valutazione
    
    # Liste per raccogliere predizioni e etichette vere
    all_preds_num = []      # Predizioni dei numeri
    all_labels_num = []     # Etichette vere dei numeri
    all_preds_char = []     # Predizioni delle lettere
    all_labels_char = []    # Etichette vere delle lettere
    
    # Disabilita i gradienti (non servono in valutazione, risparmia memoria)
    with torch.no_grad():
        for inputs, labels_num, labels_char in dataloader:
            inputs = inputs.to(device)
            labels_num = labels_num.to(device)
            labels_char = labels_char.to(device)
            
            # Forward: ottieni le predizioni delle due teste
            out_num, out_char = model(inputs)
            
            # Prendi la classe con probabilità massima
            _, preds_num = torch.max(out_num, 1)
            _, preds_char = torch.max(out_char, 1)
            
            # Salva i risultati (spostati su CPU per usarli con sklearn)
            all_preds_num.extend(preds_num.cpu().numpy())
            all_labels_num.extend(labels_num.cpu().numpy())
            all_preds_char.extend(preds_char.cpu().numpy())
            all_labels_char.extend(labels_char.cpu().numpy())
            
    return (np.array(all_labels_num), np.array(all_preds_num)), \
           (np.array(all_labels_char), np.array(all_preds_char))

print("Calcolo metriche sul Test Set...")
# Esegui la valutazione e ottieni le predizioni per numeri e lettere
(y_true_num, y_pred_num), (y_true_char, y_pred_char) = evaluate_multi_head(model, dataloaders['test'])

# Nomi delle classi per i report leggibili
names_num = [str(i) for i in range(1, 11)]  # ['1', '2', ..., '10']
names_char = ['A', 'B', 'C', 'D']          # I 4 semi

# --- REPORT NUMERI: precision, recall, F1 per ogni valore (1-10) ---
print("\n=== REPORT NUMERI ===")
print(classification_report(y_true_num, y_pred_num, target_names=names_num))

# Confusion Matrix per i numeri: matrice 10x10
plt.figure(figsize=(10, 8))
cm_num = confusion_matrix(y_true_num, y_pred_num)
sns.heatmap(cm_num, annot=True, fmt='d', cmap='Blues', 
            xticklabels=names_num, yticklabels=names_num)
plt.title('Confusion Matrix: Numeri (1-10)')
plt.ylabel('Vero')      # Asse Y: valore vero
plt.xlabel('Predetto')  # Asse X: valore predetto
plt.show()

# --- REPORT LETTERE: precision, recall, F1 per ogni seme (A-D) ---
print("\n=== REPORT LETTERE ===")
print(classification_report(y_true_char, y_pred_char, target_names=names_char))

# Confusion Matrix per le lettere: matrice 4x4
plt.figure(figsize=(6, 5))
cm_char = confusion_matrix(y_true_char, y_pred_char)
sns.heatmap(cm_char, annot=True, fmt='d', cmap='Greens', 
            xticklabels=names_char, yticklabels=names_char)
plt.title('Confusion Matrix: Lettere (A-D)')
plt.ylabel('Vero')
plt.xlabel('Predetto')
plt.show()

# ==========================================
# 7. GENERAZIONE MATRICE DI CONFUSIONE 40x40
# ==========================================
# Le due teste predicono separatamente numero e lettera.
# Per costruire la confusion matrix delle 40 carte originali,
# dobbiamo RICOMBINARE le due predizioni in una sola classe.
# Es: predizione num=9 + char=0 → "10A" → indice della classe originale

print("\nRicostruzione delle 40 classi originali per la Matrice Completa...")

# Lista delle classi originali dal dataset (es. ['10 A', '10 B', ..., '1 A', '1 B', ...])
# NOTA: ImageFolder ordina alfabeticamente, quindi "10" viene prima di "1"
original_classes = image_datasets['test'].classes
# Mappa: nome classe → indice (es. "10 A" → 0, "10 B" → 1, ...)
class_to_idx = {cls_name: i for i, cls_name in enumerate(original_classes)}

# Mappa inversa per le lettere: indice → lettera
idx_to_char = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

y_pred_combined = []  # Predizioni ricostruite come indici 0-39
y_true_combined = []  # Etichette vere ricostruite come indici 0-39

# Per ogni immagine del test set, ricombina le predizioni delle due teste
for i in range(len(y_pred_num)):
    # --- RICOSTRUZIONE DELLA PREDIZIONE ---
    pred_n_str = str(y_pred_num[i] + 1)       # Indice 0→"1", indice 9→"10"
    pred_c_str = idx_to_char[y_pred_char[i]]  # Indice 0→"A", indice 3→"D"
    pred_fullname = pred_n_str + pred_c_str   # Es: "10" + "A" = "10A"
    
    # Cerca il nome ricostruito nella lista delle classi originali
    if pred_fullname in class_to_idx:
        y_pred_combined.append(class_to_idx[pred_fullname])  # Indice 0-39
    else:
        # Combinazione impossibile (non dovrebbe succedere con 10 numeri x 4 lettere)
        print(f"Attenzione: Classe predetta {pred_fullname} non trovata nel dataset originale.")
        y_pred_combined.append(-1)

    # --- RICOSTRUZIONE DELLA VERITÀ (GROUND TRUTH) ---
    true_n_str = str(y_true_num[i] + 1)
    true_c_str = idx_to_char[y_true_char[i]]
    true_fullname = true_n_str + true_c_str
    
    if true_fullname in class_to_idx:
        y_true_combined.append(class_to_idx[true_fullname])
    else:
        y_true_combined.append(-1)

# Confusion Matrix 40x40: mostra come il modello si comporta su tutte le 40 carte
cm_full = confusion_matrix(y_true_combined, y_pred_combined)

# Visualizzazione della matrice (molto grande: 40 righe x 40 colonne)
plt.figure(figsize=(24, 20))
sns.heatmap(cm_full, annot=True, fmt='d', cmap='Purples',
            xticklabels=original_classes, 
            yticklabels=original_classes)
plt.title('Confusion Matrix Completa Ricostruita (40x40)')
plt.ylabel('Classe Vera')                                # Asse Y: carta vera
plt.xlabel('Classe Predetta (Combinazione Num+Lettera)')  # Asse X: carta predetta
plt.xticks(rotation=90)  # Ruota le etichette per leggibilità
plt.yticks(rotation=0)
plt.show()

# Report completo con precision, recall, F1 per ciascuna delle 40 carte
print("\nClassification Report Completo (40 Classi):\n")
print(classification_report(y_true_combined, y_pred_combined, target_names=original_classes))