#!/usr/bin/env python3
"""
Script per ridimensionare le immagini in foto/mazzoTopolino a 225x300 pixel
"""

from PIL import Image
import os
from pathlib import Path

def resize_images(input_folder, target_width=225, target_height=300):
    """
    Ridimensiona tutte le immagini nella cartella specificata
    
    Args:
        input_folder: Percorso della cartella con le immagini
        target_width: Larghezza target (default 225)
        target_height: Altezza target (default 300)
    """
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"Errore: la cartella {input_folder} non esiste")
        return
    
    # Estensioni immagini supportate
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    
    # Trova tutte le immagini
    images = [f for f in input_path.iterdir() 
              if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not images:
        print(f"Nessuna immagine trovata in {input_folder}")
        return
    
    print(f"Trovate {len(images)} immagini da ridimensionare")
    
    # Crea cartella di output
    output_folder = input_path / "resized"
    output_folder.mkdir(exist_ok=True)
    
    success_count = 0
    for img_path in images:
        try:
            # Apri l'immagine
            img = Image.open(img_path)
            
            # Ridimensiona mantenendo l'aspect ratio e poi crop
            img_resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # Salva l'immagine ridimensionata
            output_path = output_folder / img_path.name
            img_resized.save(output_path, quality=95)
            
            success_count += 1
            print(f"✓ Ridimensionata: {img_path.name}")
            
        except Exception as e:
            print(f"✗ Errore con {img_path.name}: {e}")
    
    print(f"\nCompletato! {success_count}/{len(images)} immagini ridimensionate con successo")
    print(f"Le immagini ridimensionate sono in: {output_folder}")

if __name__ == "__main__":
    # Percorso della cartella mazzoTopolino
    folder_path = "foto/mazzoTopolino"
    
    print("=== Ridimensionamento Immagini ===")
    print(f"Dimensione target: 225x300 pixel")
    print(f"Cartella input: {folder_path}\n")
    
    resize_images(folder_path)