import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Iterator
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold

# =============================================================================
# 1. IL PYTORCH DATASET (Caricamento Lazy & Trasformazioni Online)
# =============================================================================
class MosquitoDataset(Dataset):
    """
    Classe PyTorch standard per il caricamento lazy delle immagini.
    Applica augmentation e normalizzazione on-the-fly tramite 'transforms'.
    """
    def __init__(self, image_paths: List[Path], labels: List[int], transforms=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = str(self.image_paths[idx])
        
        # Lettura immagine: convertiamo in RGB per PyTorch
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Immagine non trovata o corrotta: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Applica l'online preprocessing (Albumentations: Augmentation + Normalizzazione)
        if self.transforms is not None:
            augmented = self.transforms(image=image)
            image = augmented["image"]
            
        label = self.labels[idx]
        return image, label # type: ignore 


# =============================================================================
# 2. LO SPLITTER (Riproducibilità & K-Fold)
# =============================================================================
class DataSplitter:
    """
    Classe richiamabile dalla Pipeline che si occupa ESCLUSIVAMENTE 
    di dividere i dati. Garantisce la riproducibilità leggendo il 'seed' dal config.
    """
    def __init__(self, config: Dict[str, Any]):
        # Se il main non passa il seed, usiamo 42 come default di sicurezza
        self.seed = config.get("seed", 42)

    def train_test_split(
        self, paths: List[Path], labels: List[int], test_size: float = 0.2
    ) -> Tuple[List[Path], List[Path], List[int], List[int]]:
        """
        Divide un set completo in due porzioni stratificate.
        Utile per separare l'Hold-Out Test Set iniziale, o per un classico Train/Val.
        """
        p_train, p_test, l_train, l_test = train_test_split(
            paths, labels,
            test_size=test_size,
            random_state=self.seed,
            stratify=labels
        )
        return p_train, p_test, l_train, l_test

    def k_fold_split(
        self, paths: List[Path], labels: List[int], k: int = 5
    ) -> Iterator[Tuple[List[Path], List[Path], List[int], List[int]]]:
        """
        Generatore per la Stratified K-Fold Cross Validation.
        Ad ogni iterazione restituisce: (train_paths, val_paths, train_labels, val_labels).
        """
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.seed)
        
        # StratifiedKFold lavora con array NumPy per l'indicizzazione
        paths_arr = np.array(paths)
        labels_arr = np.array(labels)

        for train_idx, val_idx in skf.split(paths_arr, labels_arr):
            yield (
                paths_arr[train_idx].tolist(),
                paths_arr[val_idx].tolist(),
                labels_arr[train_idx].tolist(),
                labels_arr[val_idx].tolist()
            )


# =============================================================================
# 3. IL FINDER (Ricerca su disco e Filtraggio Dataset)
# =============================================================================
class DataFinder:
    """
    Si occupa di navigare la directory pre-processata e filtrare i file
    in base ai dataset consentiti dal file di configurazione.
    """
    def __init__(self, config: Dict[str, Any]):
        self.data_path = Path(config["data_path"])
        # Set dei dataset da cui vogliamo estrarre i dati (es. {"bioscan", "mendeley"})
        self.allowed_datasets = set([ds.lower() for ds in config.get("allowed_datasets", [])])
        self.classes = config.get("classes", ["aedes", "anopheles", "culex", "non_zanzare"])
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def gather_data(self) -> Tuple[List[Path], List[int]]:
        """Restituisce le liste grezze accoppiate di tutti i percorsi e tutte le label."""
        image_paths = []
        labels = []

        if not self.data_path.exists():
            raise FileNotFoundError(f"Directory {self.data_path} non trovata!")

        for cls_name in self.classes:
            cls_dir = self.data_path / cls_name
            if not cls_dir.is_dir():
                print(f"[AVVISO] Cartella classe mancante: {cls_dir}")
                continue
                
            class_idx = self.class_to_idx[cls_name]
            
            for img_path in cls_dir.iterdir():
                if not img_path.is_file() or img_path.suffix.lower() not in {'.jpg', '.png', '.jpeg'}:
                    continue
                
                dataset_prefix = img_path.name.split("__")[0].lower()
                
                # Applica il filtro: ignora se non è nella lista dei consentiti
                if self.allowed_datasets and dataset_prefix not in self.allowed_datasets:
                    continue
                    
                image_paths.append(img_path)
                labels.append(class_idx)
                
        print(f"✅ Trovate {len(image_paths)} immagini (Filtro dataset: {self.allowed_datasets})")
        return image_paths, labels