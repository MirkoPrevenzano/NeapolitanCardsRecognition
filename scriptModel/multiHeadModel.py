# Addestramento multi-head tentativo personale resnet18
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time
import copy
import re

# ==========================================
# 0. CARICAMENTO CONFIGURAZIONE DA FILE
# ==========================================

def load_config(config_path):
    """Carica la configurazione da un file di testo key = value."""
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.split('#')[0].strip()  # rimuovi commenti inline
            config[key] = value
    return config


def parse_bool(value):
    """Converte stringa in booleano."""
    return value.lower() in ('true', '1', 'yes', 'si')


def parse_float_list(value):
    """Converte stringa '0.1, 0.2, 0.3' in lista di float."""
    return [float(x.strip()) for x in value.split(',')]


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.txt')
cfg = load_config(CONFIG_PATH)
print(f"Configurazione caricata da: {CONFIG_PATH}")

# ==========================================
# 1. CONFIGURAZIONE E IPERPARAMETRI (da config.txt)
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo utilizzato: {device}")

DATA_DIR = cfg.get('DATA_DIR', '/kaggle/input/datasets/mirkoprevenzano/versione3/datasetVersione3')

MODEL_NAME = cfg.get('MODEL', 'resnet18').lower()
BATCH_SIZE = int(cfg.get('BATCH_SIZE', '32'))
EPOCHS = int(cfg.get('EPOCHS', '30'))
LEARNING_RATE = float(cfg.get('LEARNING_RATE', '1e-4'))

USE_EARLY_STOPPING = parse_bool(cfg.get('EARLY_STOPPING', 'true'))
EARLY_STOPPING_PATIENCE = int(cfg.get('EARLY_STOPPING_PATIENCE', '6'))
EARLY_STOPPING_MIN_DELTA = float(cfg.get('EARLY_STOPPING_MIN_DELTA', '1e-4'))

USE_SCHEDULER = parse_bool(cfg.get('SCHEDULER', 'true'))
SCHEDULER_FACTOR = float(cfg.get('SCHEDULER_FACTOR', '0.1'))
SCHEDULER_PATIENCE = int(cfg.get('SCHEDULER_PATIENCE', '2'))

UNFREEZE_LAYER4 = parse_bool(cfg.get('UNFREEZE_LAYER4', 'true'))

NUM_CLASSES_NUMBER = int(cfg.get('NUM_CLASSES_NUMBER', '10'))
NUM_CLASSES_LETTER = int(cfg.get('NUM_CLASSES_LETTER', '4'))

MEAN = parse_float_list(cfg.get('MEAN', '0.6055278, 0.57453782, 0.51603037'))
STD = parse_float_list(cfg.get('STD', '0.25722615, 0.27000131, 0.28853789'))

print(f"Modello: {MODEL_NAME} | Batch: {BATCH_SIZE} | Epoche: {EPOCHS} | LR: {LEARNING_RATE}")
print(f"Early Stopping: {USE_EARLY_STOPPING} | Scheduler: {USE_SCHEDULER} | Unfreeze Layer4: {UNFREEZE_LAYER4}")


# ==========================================
# 2. DATASET PERSONALIZZATO MULTI-LABEL
# ==========================================

class MultiHeadDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # Carica dataset usando ImageFolder per sfruttare la struttura delle cartelle
        self.dataset = datasets.ImageFolder(root_dir, transform=transform)
        self.classes = self.dataset.classes
        self.letter_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

        self.img_labels = []
        # Estrazione delle etichette numeriche e letterali dalle cartelle
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


# Converto i dati in tensori per una migliore gestione durante il training con la GPU.
# Applico standardizzazione usando la media e deviazione standard calcolate in precedenza (da calcoloMediaVarianza.py).
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


TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid') 
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Creo i dataset e dataloader per ogni split
image_datasets = {
    'train': MultiHeadDataset(TRAIN_DIR, data_transforms['train']),
    'valid': MultiHeadDataset(VALID_DIR, data_transforms['valid']),
    'test': MultiHeadDataset(TEST_DIR, data_transforms['test'])
}

# Dataloader con shuffle solo per il training così da non associare l'ordine dei dati a pattern di apprendimento indesiderati, e num_workers=2 per un caricamento più efficiente.
# Applico approccio mini batch
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
    'valid': DataLoader(image_datasets['valid'], batch_size=BATCH_SIZE, shuffle=False, num_workers=2),  # [AGGIUNTA]
    'test': DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
print(f"Dataset sizes: {dataset_sizes}")

# ==========================================
# 3. MODELLO MULTI-HEAD (ResNet Configurabile)
# ==========================================


class MultiHeadResNet(nn.Module):
    def __init__(self, num_classes_num, num_classes_char, model_name='resnet18', unfreeze_layer4=True):
        super(MultiHeadResNet, self).__init__()

        # Configurazione dinamica del backbone (ResNet18 o ResNet50) con pesi pre-addestrati su ImageNet
        if model_name == 'resnet50':
            self.base_model = models.resnet50(weights="IMAGENET1K_V1")
        else:
            self.base_model = models.resnet18(weights="IMAGENET1K_V1")

        # Rimuovo la testa originale (fully connected) per sostituirla con due teste separate per numeri e lettere
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()

        # Freeze tutti i parametri del backbone
        for p in self.base_model.parameters():
            p.requires_grad = False

        # [CONFIGURABILE] unfreeze solo layer4
        if unfreeze_layer4:
            for p in self.base_model.layer4.parameters():
                p.requires_grad = True

        self.head_num = nn.Linear(num_ftrs, num_classes_num)
        self.head_char = nn.Linear(num_ftrs, num_classes_char)

        # Assicuro che le teste siano sempre trainabilI
        for p in self.head_num.parameters():
            p.requires_grad = True
        for p in self.head_char.parameters():
            p.requires_grad = True

    
    def forward(self, x):
        x = self.base_model(x)
        out_num = self.head_num(x)
        out_char = self.head_char(x)
        return out_num, out_char

model = MultiHeadResNet(
    NUM_CLASSES_NUMBER, NUM_CLASSES_LETTER,
    model_name=MODEL_NAME, unfreeze_layer4=UNFREEZE_LAYER4
).to(device)

# Configuro la loss function e l'ottimizzatore (Adam) considerando solo i parametri trainabili (teste + layer4 se unfreeze)
criterion = nn.CrossEntropyLoss()
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE)

# Applico scheduler se configurato, monitorando la val_loss per adattare dinamicamente il learning rate durante il training
if USE_SCHEDULER:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )
else:
    scheduler = None


# ==========================================
# 4. LOOP DI TRAINING MULTI-HEAD
# ==========================================

def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    num_epochs=10,
    early_stopping_patience=6,
    min_delta=1e-4,
    use_early_stopping=True,
    use_scheduler=True
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc_avg = 0.0
    best_val_loss = float('inf')
    no_improve_epochs = 0

    history = {
        'train_loss': [], 'valid_loss': [],
        'train_acc_num': [], 'valid_acc_num': [],
        'train_acc_char': [], 'valid_acc_char': [],
        'lr': []
    }
    # Inizio iterazione sulle epoche
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Iterazione sulle fasi: train e validation
        for phase in ['train', 'valid']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects_num = 0
            running_corrects_char = 0
            # Iterazione sui batch di dati
            for inputs, labels_num, labels_char in dataloaders[phase]:
                inputs = inputs.to(device)
                labels_num = labels_num.to(device)
                labels_char = labels_char.to(device)
                # Azzeramento dei gradienti prima del backward pass
                optimizer.zero_grad()

                # se siamo in fase di training, abilitiamo il calcolo dei gradienti, altrimenti no.
                with torch.set_grad_enabled(phase == 'train'):
                    outputs_num, outputs_char = model(inputs)

                    # Calcolo delle predizioni e delle perdite per entrambe le teste
                    _, preds_num = torch.max(outputs_num, 1)
                    _, preds_char = torch.max(outputs_char, 1)

                    loss_num = criterion(outputs_num, labels_num)
                    loss_char = criterion(outputs_char, labels_char)
                    loss = loss_num + loss_char

                    if phase == 'train':
                        # Solo durante il training eseguiamo il backward pass e l'aggiornamento dei pesi
                        loss.backward()
                        optimizer.step()

                # Aggiornamento delle metriche di perdita e accuratezza per entrambe le teste
                running_loss += loss.item() * inputs.size(0)
                running_corrects_num += torch.sum(preds_num == labels_num.data)
                running_corrects_char += torch.sum(preds_char == labels_char.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc_num = running_corrects_num.double() / dataset_sizes[phase]
            epoch_acc_char = running_corrects_char.double() / dataset_sizes[phase]
            epoch_acc_avg = (epoch_acc_num + epoch_acc_char) / 2

            print(f'{phase} Loss: {epoch_loss:.4f} | Acc Num: {epoch_acc_num:.4f} | Acc Char: {epoch_acc_char:.4f}')

            # Salvataggio delle metriche per visualizzazione successiva
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc_num'].append(epoch_acc_num.item())
                history['train_acc_char'].append(epoch_acc_char.item())
            else:
                history['valid_loss'].append(epoch_loss)
                history['valid_acc_num'].append(epoch_acc_num.item())
                history['valid_acc_char'].append(epoch_acc_char.item())

                # Aggiornamento dello scheduler se configurato, monitorando la val_loss per adattare dinamicamente il learning rate durante il training
                if use_scheduler and scheduler is not None:
                    scheduler.step(epoch_loss)
                current_lr = optimizer.param_groups[0]['lr']
                history['lr'].append(current_lr)
                print(f'LR corrente: {current_lr:.6f}')

                # miglior modello su acc media
                if epoch_acc_avg > best_acc_avg:
                    best_acc_avg = epoch_acc_avg
                    best_model_wts = copy.deepcopy(model.state_dict())

                # early stopping basato su val_loss
                if use_early_stopping:
                    if epoch_loss < (best_val_loss - min_delta):
                        best_val_loss = epoch_loss
                        no_improve_epochs = 0
                    else:
                        no_improve_epochs += 1

        print()

        if use_early_stopping and no_improve_epochs >= early_stopping_patience:
            print(f"[Early Stopping] Nessun miglioramento in val_loss per {early_stopping_patience} epoche.")
            break

    time_elapsed = time.time() - since
    print(f'Training completato in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Miglior Avg Val Acc: {best_acc_avg:.4f}')

    model.load_state_dict(best_model_wts)
    return model, history

model, history = train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    num_epochs=EPOCHS,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    min_delta=EARLY_STOPPING_MIN_DELTA,
    use_early_stopping=USE_EARLY_STOPPING,
    use_scheduler=USE_SCHEDULER
)

# ==========================================
# 5. VISUALIZZAZIONE TRAINING
# ==========================================

plt.figure(figsize=(18, 5))

# Loss
plt.subplot(1, 3, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['valid_loss'], label='Valid Loss')
plt.title('Loss Totale')
plt.legend()

# Acc Numeri
plt.subplot(1, 3, 2)
plt.plot(history['train_acc_num'], label='Train Num Acc')
plt.plot(history['valid_acc_num'], label='Valid Num Acc')
plt.title('Accuratezza Numeri (1-10)')
plt.legend()

# Acc Lettere
plt.subplot(1, 3, 3)
plt.plot(history['train_acc_char'], label='Train Char Acc')
plt.plot(history['valid_acc_char'], label='Valid Char Acc')
plt.title('Accuratezza Lettere (A-D)')
plt.legend()

plt.show()

# ==========================================
# 6. VALUTAZIONE SUL TEST SET & METRICHE
# ==========================================

def evaluate_multi_head(model, dataloader):
    model.eval()
    
    all_preds_num = []
    all_labels_num = []
    all_preds_char = []
    all_labels_char = []
    
    with torch.no_grad():
        for inputs, labels_num, labels_char in dataloader:
            inputs = inputs.to(device)
            labels_num = labels_num.to(device)
            labels_char = labels_char.to(device)
            
            out_num, out_char = model(inputs)
            
            _, preds_num = torch.max(out_num, 1)
            _, preds_char = torch.max(out_char, 1)
            
            all_preds_num.extend(preds_num.cpu().numpy())
            all_labels_num.extend(labels_num.cpu().numpy())
            all_preds_char.extend(preds_char.cpu().numpy())
            all_labels_char.extend(labels_char.cpu().numpy())
            
    return (np.array(all_labels_num), np.array(all_preds_num)), \
           (np.array(all_labels_char), np.array(all_preds_char))

print("Calcolo metriche sul Test Set...")
(y_true_num, y_pred_num), (y_true_char, y_pred_char) = evaluate_multi_head(model, dataloaders['test'])

# Nomi classi per visualizzazione
names_num = [str(i) for i in range(1, 11)] # '1'...'10'
names_char = ['A', 'B', 'C', 'D']

# --- REPORT NUMERI ---
print("\n=== REPORT NUMERI ===")
print(classification_report(y_true_num, y_pred_num, target_names=names_num))

# Confusion Matrix Numeri
plt.figure(figsize=(10, 8))
cm_num = confusion_matrix(y_true_num, y_pred_num)
sns.heatmap(cm_num, annot=True, fmt='d', cmap='Blues', 
            xticklabels=names_num, yticklabels=names_num)
plt.title('Confusion Matrix: Numeri (1-10)')
plt.ylabel('Vero')
plt.xlabel('Predetto')
plt.show()

# --- REPORT LETTERE ---
print("\n=== REPORT LETTERE ===")
print(classification_report(y_true_char, y_pred_char, target_names=names_char))

# Confusion Matrix Lettere
plt.figure(figsize=(6, 5))
cm_char = confusion_matrix(y_true_char, y_pred_char)
sns.heatmap(cm_char, annot=True, fmt='d', cmap='Greens', 
            xticklabels=names_char, yticklabels=names_char)
plt.title('Confusion Matrix: Lettere (A-D)')
plt.ylabel('Vero')
plt.xlabel('Predetto')
plt.show()

# ==========================================
# 7. GENERAZIONE MATRICE DI CONFUSIONE 40x40 (FIXED)
# ==========================================

print("\nGenerazione Matrice di Confusione Completa (40 Classi)...")

# --- MODIFICA APPLICATA (FIX) ---
# Ricostruzione intelligente basata su mappa inversa (Pair -> Folder Name)
# Questo risolve il problema se le cartelle non si chiamano esattamente "10A" (es. "10_A")

original_classes = image_datasets['test'].classes
class_to_idx = {cls_name: i for i, cls_name in enumerate(original_classes)}

# Regex e mappa per ricostruire l'associazione
regex_pattern = re.compile(r"(\d+)\s*([A-Z])")
letter_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

# Creiamo una mappa dinamica: Tuple(NumIdx, CharIdx) -> NomeCartellaOriginale
pair_to_original_name = {}

for cls_name in original_classes:
    match = regex_pattern.search(cls_name)
    if match:
        n_str, c_str = match.groups()
        n_idx = int(n_str) - 1  # 1->0
        c_idx = letter_map[c_str] # A->0
        
        # Mappa la coppia di indici (es: 9, 0) al nome reale della cartella (es: "10_A")
        pair_to_original_name[(n_idx, c_idx)] = cls_name

y_pred_combined = []
y_true_combined = []

# Iteriamo sulle predizioni grezze (indici)
for i in range(len(y_pred_num)):
    
    # 1. Recupera la predizione combinata
    pred_pair = (y_pred_num[i], y_pred_char[i])
    
    if pred_pair in pair_to_original_name:
        # Troviamo il nome della cartella corrispondente a questa combinazione numero/lettera
        pred_name_real = pair_to_original_name[pred_pair]
        y_pred_combined.append(class_to_idx[pred_name_real])
    else:
        # Caso: Il modello ha predetto una combinazione che non esiste nel dataset (es. 11 A se non c'è)
        # O se la mappatura ha fallito. Mettiamo -1 per escluderlo o gestirlo.
        y_pred_combined.append(-1)

    # 2. Recupera la classe vera combinata
    true_pair = (y_true_num[i], y_true_char[i])
    
    if true_pair in pair_to_original_name:
        true_name_real = pair_to_original_name[true_pair]
        y_true_combined.append(class_to_idx[true_name_real])
    else:
        y_true_combined.append(-1)

# Filtriamo eventuali errori (-1)
valid_mask = [i for i, x in enumerate(y_pred_combined) if x != -1 and y_true_combined[i] != -1]
y_pred_clean = [y_pred_combined[i] for i in valid_mask]
y_true_clean = [y_true_combined[i] for i in valid_mask]

# Visualizzazione Matrice Completa
if len(y_true_clean) > 0:
    plt.figure(figsize=(24, 20))
    cm_full = confusion_matrix(y_true_clean, y_pred_clean)
    
    sns.heatmap(cm_full, annot=True, fmt='d', cmap='Purples',
                xticklabels=original_classes, 
                yticklabels=original_classes)

    plt.title('Confusion Matrix Completa Ricostruita (40x40)')
    plt.ylabel('Classe Vera')
    plt.xlabel('Classe Predetta')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()

    # Classification Report Completo
    print("\nClassification Report Completo (40 Classi):\n")
    print(classification_report(y_true_clean, y_pred_clean, target_names=original_classes, labels=range(len(original_classes))))
else:
    print("ERRORE CRITICO: Non è stato possibile ricostruire nessuna classe. Verifica i nomi delle cartelle.")