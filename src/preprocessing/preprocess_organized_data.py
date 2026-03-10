"""
Preprocessing Offline per MOSQ-AI.
Applica: CLAHE, Letterboxing (pad to square) e Lanczos Resizing (224x224).
Salva in un formato standardizzato (RGB, JPG alta qualità).

Uso:
    conda activate mosq_env
    python src/data_prep/preprocess_offline.py
"""

import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data" / "organized_raw"
OUTPUT_DIR = ROOT / "data" / "preprocessed"

# Standard ImageNet
TARGET_SIZE = 224
# Colore di padding neutro (grigio medio, non altera la luminosità media)
PAD_COLOR = (128, 128, 128) 
CLASSES = ["aedes", "anopheles", "culex", "non_zanzare"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ---------------------------------------------------------------------------
# Funzione Core di Preprocessing (I 4 Step)
# ---------------------------------------------------------------------------
def process_single_image(input_path: Path, output_path: Path) -> bool:
    try:
        # 1. Lettura e Standardizzazione Formato (RGB)
        # cv2 legge in BGR di default.
        img_bgr = cv2.imread(str(input_path))
        if img_bgr is None:
            return False
        
        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Convertiamo in spazio colore LAB per applicare CLAHE solo alla luminosità (L),
        # evitando di sfasare i colori (canali A e B).
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        
        lab_clahe = cv2.merge((cl, a_channel, b_channel))
        img_clahe_bgr = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        # 3. Letterboxing e Lanczos Resizing
        h, w = img_clahe_bgr.shape[:2]
        
        # Calcola la scala per far rientrare il lato più lungo nel TARGET_SIZE
        scale = min(TARGET_SIZE / h, TARGET_SIZE / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize SOTA per minimizzare perdita di texture negli insetti
        resized = cv2.resize(img_clahe_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Crea una tela quadrata vuota del colore di padding
        canvas = np.full((TARGET_SIZE, TARGET_SIZE, 3), PAD_COLOR, dtype=np.uint8)
        
        # Calcola gli offset per centrare l'immagine ridimensionata
        y_offset = (TARGET_SIZE - new_h) // 2
        x_offset = (TARGET_SIZE - new_w) // 2
        
        # Incolla l'immagine al centro
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # 4. Salvataggio Standard (JPG ad alta qualità)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Usiamo il parametro di qualità JPEG al 95%
        cv2.imwrite(str(output_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        return True
    
    except Exception as e:
        print(f"Errore su {input_path.name}: {e}")
        return False

# ---------------------------------------------------------------------------
# Orchestrazione Multiprocessing
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(f"Inizio Preprocessing Offline SOTA ({TARGET_SIZE}x{TARGET_SIZE})")
    print("=" * 60)
    
    tasks = []
    # Raccogliamo tutti i percorsi delle immagini
    for cls in CLASSES:
        cls_dir = INPUT_DIR / cls
        if not cls_dir.exists():
            continue
            
        for img_path in cls_dir.iterdir():
            if img_path.suffix.lower() in VALID_EXTENSIONS:
                out_path = OUTPUT_DIR / cls / f"{img_path.stem}.jpg"
                tasks.append((img_path, out_path))
                
    if not tasks:
        print("Nessuna immagine trovata in data/organized_raw/")
        return

    print(f"Trovate {len(tasks):,} immagini. Avvio elaborazione in parallelo...")
    
    success_count = 0
    # ProcessPoolExecutor è molto più veloce di ThreadPool per operazioni CPU-bound (OpenCV)
    with ProcessPoolExecutor() as executor:
        # Sottomissione task
        futures = {executor.submit(process_single_image, inp, out): inp for inp, out in tasks}
        
        # Barra di progresso
        with tqdm(total=len(tasks), desc="Preprocessing", unit="img") as bar:
            for future in as_completed(futures):
                if future.result():
                    success_count += 1
                bar.update(1)
                
    print("\n" + "=" * 60)
    print(f"Completato! Salvate {success_count:,} immagini in: {OUTPUT_DIR}")
    print("Ora siamo pronti per PyTorch!")

if __name__ == "__main__":
    main()