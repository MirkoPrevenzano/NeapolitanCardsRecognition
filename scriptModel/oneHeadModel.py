# Addestramento su 40 classi (one-head)
# Note personali: qui non uso weight decay (vedi configurazione ottimizzatore).

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time
import copy

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


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'oneHeadConfig.txt')
cfg = load_config(CONFIG_PATH)
print(f"Configurazione caricata da: {CONFIG_PATH}")

# ==========================================
# 1. CONFIGURAZIONE E IPERPARAMETRI (da oneHeadConfig.txt)
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo utilizzato: {device}")

DATA_DIR = cfg.get('DATA_DIR', '/kaggle/input/datasets/mirkoprevenzano/versione3/datasetVersione3')

MODEL_NAME = cfg.get('MODEL', 'resnet18').lower()
BATCH_SIZE = int(cfg.get('BATCH_SIZE', '32'))
NUM_CLASSES = int(cfg.get('NUM_CLASSES', '40'))
EPOCHS = int(cfg.get('EPOCHS', '30'))
LEARNING_RATE = float(cfg.get('LEARNING_RATE', '1e-4'))

USE_EARLY_STOPPING = parse_bool(cfg.get('EARLY_STOPPING', 'true'))
EARLY_STOPPING_PATIENCE = int(cfg.get('EARLY_STOPPING_PATIENCE', '6'))
EARLY_STOPPING_MIN_DELTA = float(cfg.get('EARLY_STOPPING_MIN_DELTA', '1e-4'))

USE_SCHEDULER = parse_bool(cfg.get('SCHEDULER', 'true'))
SCHEDULER_FACTOR = float(cfg.get('SCHEDULER_FACTOR', '0.1'))
SCHEDULER_PATIENCE = int(cfg.get('SCHEDULER_PATIENCE', '2'))

UNFREEZE_LAYER4 = parse_bool(cfg.get('UNFREEZE_LAYER4', 'true'))

MEAN = parse_float_list(cfg.get('MEAN', '0.6055278, 0.57453782, 0.51603037'))
STD = parse_float_list(cfg.get('STD', '0.25722615, 0.27000131, 0.28853789'))

print(f"Modello: {MODEL_NAME} | Batch: {BATCH_SIZE} | Epoche: {EPOCHS} | LR: {LEARNING_RATE}")
print(f"Early Stopping: {USE_EARLY_STOPPING} | Scheduler: {USE_SCHEDULER} | Unfreeze Layer4: {UNFREEZE_LAYER4}")

# ==========================================
# 2. DATASET E DATALOADERS
# ==========================================

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

image_datasets = {
    # Carico dataset usando ImageFolder per sfruttare la struttura delle cartelle (una cartella = una classe)
    'train': datasets.ImageFolder(TRAIN_DIR, data_transforms['train']),
    'valid': datasets.ImageFolder(VALID_DIR, data_transforms['valid']),
    'test': datasets.ImageFolder(TEST_DIR, data_transforms['test'])
}

# Dataloader con shuffle solo per il training così da non associare l'ordine dei dati a pattern di apprendimento indesiderati,
# e num_workers=2 per un caricamento più efficiente.
# Applico approccio mini batch.
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
    'valid': DataLoader(image_datasets['valid'], batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
    'test': DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
class_names = image_datasets['train'].classes

print(f"Classi trovate: {len(class_names)}")
print(f"Esempio classi: {class_names[:5]}...")
print(f"Dimensione Train: {dataset_sizes['train']}, Valid: {dataset_sizes['valid']}, Test: {dataset_sizes['test']}")

# ==========================================
# 3. DEFINIZIONE DEL MODELLO (ResNet Configurabile)
# ==========================================

def initialize_model(num_classes, model_name='resnet18', unfreeze_layer4=True):
    # Configurazione dinamica del backbone (ResNet18 o ResNet50) con pesi pre-addestrati su ImageNet
    # [CONFIGURABILE] scelta architettura da config
    if model_name == 'resnet50':
        model = models.resnet50(weights="IMAGENET1K_V1")
    else:
        model = models.resnet18(weights="IMAGENET1K_V1")

    # Freeze iniziale di tutti i layer del backbone
    for param in model.parameters():
        param.requires_grad = False

    # [CONFIGURABILE] Fine-tuning: sblocco layer4
    if unfreeze_layer4:
        for param in model.layer4.parameters():
            param.requires_grad = True

    # Sostituisco la testa originale (fully connected) con una nuova FC per NUM_CLASSES.
    # La FC va sempre allenata.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    for param in model.fc.parameters():
        param.requires_grad = True

    return model

model = initialize_model(NUM_CLASSES, model_name=MODEL_NAME, unfreeze_layer4=UNFREEZE_LAYER4).to(device)

criterion = nn.CrossEntropyLoss()

# Configuro la loss function e l'ottimizzatore (Adam) considerando solo i parametri trainabili
# (FC + layer4 se unfreeze). In questo script non imposto weight decay.
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE)

# [CONFIGURABILE] Scheduler dinamico sul validation loss
if USE_SCHEDULER:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE
    )
else:
    scheduler = None

# ==========================================
# 4. LOOP DI TRAINING
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
    best_acc = 0.0
    best_val_loss = float('inf')
    no_improve_epochs = 0

    history = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': [], 'lr': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Azzeramento dei gradienti prima del backward pass
                optimizer.zero_grad()

                # Se siamo in fase di training abilitiamo i gradienti, altrimenti no
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        # Solo durante il training eseguiamo il backward pass e l'aggiornamento dei pesi
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['valid_loss'].append(epoch_loss)
                history['valid_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'valid':
                # [CONFIGURABILE] Scheduler step sul validation loss
                if use_scheduler and scheduler is not None:
                    scheduler.step(epoch_loss)

                current_lr = optimizer.param_groups[0]['lr']
                history['lr'].append(current_lr)
                print(f'LR corrente: {current_lr:.6f}')

                # Best model by validation accuracy
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                # [CONFIGURABILE] Early stopping by validation loss
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
    print(f'Miglior Val Acc: {best_acc:.4f}')

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

plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['valid_loss'], label='Valid Loss')
plt.title('Loss durante il Training')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['valid_acc'], label='Valid Acc')
plt.title('Accuratezza durante il Training')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history['lr'], label='Learning Rate')
plt.title('LR Scheduler')
plt.legend()

plt.tight_layout()
plt.show()

# ==========================================
# 6. VALUTAZIONE SUL TEST SET & METRICHE
# ==========================================

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)

print("Calcolo metriche sul Test Set...")
y_true, y_pred = evaluate_model(model, dataloaders['test'])

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(20, 16))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Vero')
plt.xlabel('Predetto')
plt.title('Confusion Matrix')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()