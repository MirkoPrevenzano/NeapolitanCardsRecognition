# ==========================================================
# model.py
# Scopo: addestrare un modello ResNet50 (pre-addestrato su
#        ImageNet) per classificare 40 carte napoletane.
#        Una singola testa di output predice direttamente
#        una delle 40 classi (es. "1 A", "10 D", ecc.).
# ==========================================================

import os                           # Per gestire percorsi di file e cartelle
import torch                        # PyTorch: framework principale per reti neurali
import torch.nn as nn               # Moduli per costruire strati della rete (Linear, Conv2d, ecc.)
import torch.optim as optim         # Ottimizzatori (Adam, SGD, ecc.) per aggiornare i pesi
from torch.utils.data import DataLoader  # Carica i dati in batch (gruppi) durante il training
from torchvision import datasets, models, transforms  # Dataset da cartelle, modelli pre-addestrati, trasformazioni immagini
import numpy as np                  # Calcoli numerici su array
import matplotlib.pyplot as plt     # Creazione di grafici (loss, accuratezza)
import seaborn as sns               # Grafici statistici avanzati (heatmap per confusion matrix)
from sklearn.metrics import confusion_matrix, classification_report  # Metriche di valutazione del modello
import time                         # Per misurare il tempo di training
import copy                         # Per copiare in profondità i pesi del miglior modello

# ==========================================
# 1. CONFIGURAZIONE E IPERPARAMETRI
# ==========================================

# Sceglie automaticamente se usare la GPU (molto più veloce) o la CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo utilizzato: {device}")

# Cartella principale del dataset (contiene le sottocartelle train, valid, test)
DATA_DIR = '/kaggle/input/augmented/augmentationDataset'

# --- Iperparametri ---
BATCH_SIZE = 64         # Quante immagini vengono elaborate insieme ad ogni passo
NUM_CLASSES = 40        # Numero totale di classi (10 valori x 4 semi = 40 carte)
EPOCHS = 30             # Quante volte il modello vede l'intero dataset di training
LEARNING_RATE = 0.0001  # Velocità di apprendimento: quanto i pesi cambiano ad ogni passo

# Media e deviazione standard calcolate SOLO sul training set (vedi config.py)
# Servono per normalizzare le immagini: (pixel - mean) / std
MEAN = [0.64737422, 0.63253646, 0.59358844]
STD = [0.1556388, 0.17054404, 0.19233133]

# ==========================================
# 2. DATASET E DATALOADERS
# ==========================================

# Trasformazioni da applicare alle immagini prima di darle al modello:
# 1. ToTensor(): converte l'immagine da formato PIL (0-255) a tensore PyTorch (0.0-1.0)
# 2. Normalize(): normalizza ogni canale RGB con media e std del training set
# NOTA: le stesse trasformazioni vengono applicate a train, valid e test
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),              # Converte immagine in tensore
        transforms.Normalize(MEAN, STD)     # Normalizza: (pixel - media) / std
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
TRAIN_DIR = os.path.join(DATA_DIR, 'train')   # Cartella delle immagini di addestramento
VALID_DIR = os.path.join(DATA_DIR, 'valid')   # Cartella delle immagini di validazione
TEST_DIR = os.path.join(DATA_DIR, 'test')     # Cartella delle immagini di test

# ImageFolder: carica le immagini dalle cartelle e assegna automaticamente
# le etichette in base al NOME DELLA SOTTOCARTELLA (es. "1 A", "10 D")
# Il modello NON vede mai il nome del file, solo i pixel dell'immagine
image_datasets = {
    'train': datasets.ImageFolder(TRAIN_DIR, data_transforms['train']),
    'valid': datasets.ImageFolder(VALID_DIR, data_transforms['valid']),
    'test': datasets.ImageFolder(TEST_DIR, data_transforms['test'])
}

# DataLoader: raggruppa le immagini in batch e le fornisce al modello
# shuffle=True: mescola i dati ad ogni epoca (importante per il training)
# num_workers=2: usa 2 processi paralleli per caricare le immagini più velocemente
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
    'valid': DataLoader(image_datasets['valid'], batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
    'test': DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
}

# Conta quante immagini ci sono in ogni split
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
# Lista dei nomi delle classi (es. ['1 A', '1 B', ..., '10 D'])
class_names = image_datasets['train'].classes

print(f"Classi trovate: {len(class_names)}")
print(f"Esempio classi: {class_names[:5]}...")
print(f"Dimensione Train: {dataset_sizes['train']}, Valid: {dataset_sizes['valid']}, Test: {dataset_sizes['test']}")

# ==========================================
# 3. DEFINIZIONE DEL MODELLO (ResNet50)
# ==========================================

def initialize_model(num_classes):
    # Carica ResNet50 già addestrata su ImageNet (1000 classi di oggetti generici)
    # I pesi pre-addestrati permettono al modello di partire con una buona
    # conoscenza di forme, colori e texture, senza imparare tutto da zero
    model = models.resnet50(weights='IMAGENET1K_V1')
        
    # L'ultimo strato di ResNet50 ha 1000 uscite (classi ImageNet).
    # Lo sostituiamo con uno strato che ha NUM_CLASSES uscite (40 carte)
    num_ftrs = model.fc.in_features     # Numero di feature in ingresso all'ultimo strato (2048)
    model.fc = nn.Linear(num_ftrs, num_classes)  # Nuovo strato finale: 2048 -> 40
    
    return model

# Crea il modello e spostalo sulla GPU (o CPU)
model = initialize_model(NUM_CLASSES)
model = model.to(device)

# CrossEntropyLoss: funzione di errore per classificazione multi-classe
# Misura quanto le predizioni del modello sono lontane dalle etichette vere
criterion = nn.CrossEntropyLoss()

# Adam: algoritmo di ottimizzazione che aggiorna i pesi della rete
# model.fc.parameters(): ottimizziamo SOLO l'ultimo strato (fine-tuning leggero),
# mantenendo congelati tutti gli strati precedenti di ResNet50
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# ==========================================
# 4. LOOP DI TRAINING
# ==========================================

def train_model(model, criterion, optimizer, num_epochs=10):
    # Segna il tempo di inizio per calcolare la durata totale
    since = time.time()
    
    # Salva una copia dei pesi migliori trovati finora
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0  # Migliore accuratezza di validazione trovata
    
    # Dizionario per salvare lo storico di loss e accuratezza ad ogni epoca
    history = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}

    # Ciclo principale: ripeti per num_epochs volte
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Ogni epoca ha due fasi: training (impara) e validazione (valuta)
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()   # Modalità training: attiva dropout e batch normalization
            else:
                model.eval()    # Modalità valutazione: disattiva dropout

            running_loss = 0.0      # Errore accumulato nell'epoca
            running_corrects = 0    # Predizioni corrette accumulate

            # Scorri tutti i batch di immagini
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)  # Sposta le immagini sulla GPU
                labels = labels.to(device)  # Sposta le etichette sulla GPU

                optimizer.zero_grad()  # Azzera i gradienti dal passo precedente

                # Calcola i gradienti solo durante il training, non in validazione
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)          # Forward: passa le immagini nella rete
                    _, preds = torch.max(outputs, 1) # Prendi la classe con probabilità più alta
                    loss = criterion(outputs, labels) # Calcola l'errore

                    # Backward: calcola come aggiornare i pesi + aggiorna (solo in training)
                    if phase == 'train':
                        loss.backward()      # Calcola i gradienti
                        optimizer.step()     # Aggiorna i pesi della rete

                # Accumula l'errore e le predizioni corrette
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Calcola errore medio e accuratezza per questa epoca
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            # Salva i valori nello storico per i grafici
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['valid_loss'].append(epoch_loss)
                history['valid_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Se l'accuratezza di validazione è la migliore finora, salva i pesi
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    # Stampa il tempo totale di training
    time_elapsed = time.time() - since
    print(f'Training completato in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Miglior Val Acc: {best_acc:.4f}')

    # Ricarica i pesi del miglior modello (quello con la migliore validazione)
    model.load_state_dict(best_model_wts)
    return model, history

# Avvia il training del modello
model, history = train_model(model, criterion, optimizer, num_epochs=EPOCHS)

# ==========================================
# 5. VISUALIZZAZIONE TRAINING
# ==========================================

# Crea una figura con 2 grafici affiancati
plt.figure(figsize=(12, 5))

# Grafico 1: andamento dell'errore (loss) durante le epoche
plt.subplot(1, 2, 1)                                    # Primo grafico (1 riga, 2 colonne, posizione 1)
plt.plot(history['train_loss'], label='Train Loss')      # Linea dell'errore di training
plt.plot(history['valid_loss'], label='Valid Loss')      # Linea dell'errore di validazione
plt.title('Loss durante il Training')
plt.legend()

# Grafico 2: andamento dell'accuratezza durante le epoche
plt.subplot(1, 2, 2)                                    # Secondo grafico
plt.plot(history['train_acc'], label='Train Acc')        # Linea accuratezza training
plt.plot(history['valid_acc'], label='Valid Acc')        # Linea accuratezza validazione
plt.title('Accuratezza durante il Training')
plt.legend()
plt.show()  # Mostra i grafici

# ==========================================
# 6. VALUTAZIONE SUL TEST SET & METRICHE
# ==========================================

def evaluate_model(model, dataloader):
    """Valuta il modello sul test set e restituisce etichette vere e predette."""
    model.eval()        # Disattiva dropout e batch norm (modalità valutazione)
    all_preds = []      # Lista per salvare tutte le predizioni
    all_labels = []     # Lista per salvare tutte le etichette vere
    
    # torch.no_grad(): disabilita il calcolo dei gradienti (non servono in valutazione)
    # Questo risparmia memoria e velocizza l'esecuzione
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)      # Sposta immagini su GPU
            labels = labels.to(device)      # Sposta etichette su GPU
            
            outputs = model(inputs)          # Passa le immagini nella rete
            _, preds = torch.max(outputs, 1) # Prendi la classe con probabilità massima
            
            # Salva predizioni e etichette (spostate su CPU per usarle con sklearn)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds)

print("Calcolo metriche sul Test Set...")
# Ottieni le etichette vere (y_true) e le predizioni del modello (y_pred)
y_true, y_pred = evaluate_model(model, dataloaders['test'])

# Classification Report: tabella con precision, recall e F1-score per ogni classe
# - Precision: quante delle predizioni per una classe sono corrette
# - Recall: quante delle immagini reali di una classe sono state trovate
# - F1-score: media armonica di precision e recall
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix: matrice 40x40 che mostra per ogni classe
# quante immagini sono state classificate correttamente e quali errori sono stati fatti
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(20, 16))  # Dimensione grande per far entrare 40 classi
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',  # Heatmap con numeri e colori blu
            xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Vero')          # Asse Y: classe vera dell'immagine
plt.xlabel('Predetto')      # Asse X: classe predetta dal modello
plt.title('Confusion Matrix')
plt.xticks(rotation=90)     # Ruota le etichette dell'asse X per leggibilità
plt.yticks(rotation=0)
plt.show()