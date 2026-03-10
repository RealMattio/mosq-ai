import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm

class OnlinePreprocessor:
    """
    Gestisce le trasformazioni online (Data Augmentation e Normalizzazione Tensori).
    Calcola le statistiche SOLO sul train set se richiesto dal config.
    Restituisce le pipeline di trasformazione da iniettare nel PyTorch Dataset.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 'imagenet', 'custom_z_score', o 'custom_min_max'
        self.norm_strategy  = config.get("norm_strategy",  "imagenet")
        self.use_augmentation = config.get("use_augmentation", True)
        
        # Valori di default (ImageNet)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.max_pixel_value = 255.0

    def fit(self, train_paths: List[Path]) -> None:
        """
        Calcola i parametri di normalizzazione leggendo ESCLUSIVAMENTE il Train Set.
        Non carica tutto in RAM contemporaneamente, ma processa a blocchi.
        """
        if self.norm_strategy == "imagenet":
            print("[Preprocessor] Uso statistiche ImageNet predefinite. Nessun fit necessario.")
            return

        print(f"[Preprocessor] Calcolo statistiche '{self.norm_strategy}' sul Train Set...")
        
        if self.norm_strategy == "custom_min_max":
            # Per min-max standard 0-1, basta dividere per 255. 
            # I valori delle immagini a 8-bit sono sempre [0, 255].
            self.mean = [0.0, 0.0, 0.0]
            self.std = [1.0, 1.0, 1.0]
            self.max_pixel_value = 255.0
            print("[Preprocessor] Min-Max (0-1) configurato.")
            return

        if self.norm_strategy == "custom_z_score":
            # Calcolo di Mean e Std channel-wise iterativo per non saturare la RAM
            pixel_num = 0
            channel_sum = np.zeros(3)
            channel_sum_squared = np.zeros(3)

            for path in tqdm(train_paths, desc="Calcolo Mean/Std"):
                img = cv2.imread(str(path))
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0 # Normalizza 0-1 provvisoriamente
                
                pixel_num += (img.shape[0] * img.shape[1])
                # Somma lungo altezza e larghezza per ogni canale RGB
                channel_sum += np.sum(img, axis=(0, 1))
                channel_sum_squared += np.sum(np.square(img), axis=(0, 1))

            # Media = Somma / N
            self.mean = (channel_sum / pixel_num).tolist()
            # Varianza = E[X^2] - (E[X])^2
            variance = (channel_sum_squared / pixel_num) - np.square(self.mean)
            self.std = np.sqrt(variance).tolist()
            
            # max_pixel_value a 1.0 perché abbiamo già diviso in fit() e Albumentations lo aspetta
            self.max_pixel_value = 1.0 
            
            print(f"[Preprocessor] Z-Score calcolato -> Mean: {self.mean}, Std: {self.std}")


    def get_transforms(self, is_train: bool) -> A.Compose:
        """
        Restituisce la pipeline di trasformazione (Augmentation + Normalizzazione + ToTensor).
        In fase di Test/Val (is_train=False) viene applicata SOLO la normalizzazione calcolata sul Train.
        """
        transforms_list = []

        if is_train and self.use_augmentation:
            # --- DATA AUGMENTATION (Solo per il Train Set, se abilitata) ---
            # Mosse SOTA per insetti: rotazioni libere, variazioni di luce
            print("[Preprocessor] Data Augmentation: ABILITATA")
            transforms_list.extend([
                A.Rotate(limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=(128,128,128)), # type: ignore
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2)
            ])
        elif is_train:
            print("[Preprocessor] Data Augmentation: DISABILITATA")

        # --- NORMALIZZAZIONE E CONVERSIONE IN TENSORE (Per tutti) ---
        # Applica i valori ImageNet, o quelli calcolati nel fit()
        transforms_list.extend([
            A.Normalize(
                mean=self.mean,
                std=self.std,
                max_pixel_value=self.max_pixel_value,
                always_apply=True # type: ignore
            ),
            ToTensorV2() # Converte array (H, W, C) in Tensore PyTorch (C, H, W)
        ])

        return A.Compose(transforms_list)