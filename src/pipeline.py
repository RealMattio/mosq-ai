import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.data_loading.data_module import DataFinder, DataSplitter, MosquitoDataset
from src.preprocessing.online_preprocessor import OnlinePreprocessor
from src.models.mosquito_model import MosquitoNet
from src.training.trainer import Trainer
from src.evaluation.evaluator import ModelEvaluator


class Pipeline:
    """
    Il Direttore d'Orchestra.
    Riceve il dizionario di configurazione e coordina l'intero ciclo di vita
    del Machine Learning: Dati -> Preprocessing -> Modello -> Addestramento -> Valutazione.
    """
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Pipeline] Device selezionato: {self.device}")

    def run(self):
        print("\n" + "="*60)
        print("🚀 AVVIO PIPELINE MOSQ-AI 🚀")
        print("="*60)

        # Cartella di output per questo run
        run_dir = Path(self.config["output_dir"])

        # Inizializza il valutatore (la cartella viene creata solo a fine training)
        evaluator = ModelEvaluator(
            config=self.config,
            output_dir=run_dir,
            classes=self.config["classes"],
        )

        # ---------------------------------------------------------
        # 1. RICERCA E FILTRAGGIO DATI
        # ---------------------------------------------------------
        finder = DataFinder(self.config)
        all_paths, all_labels = finder.gather_data()

        # ---------------------------------------------------------
        # 2. DATA SPLIT (Train / Validation)
        # ---------------------------------------------------------
        splitter = DataSplitter(self.config)
        test_size = self.config.get("val_split", 0.2)

        train_paths, val_paths, train_labels, val_labels = splitter.train_test_split(
            all_paths, all_labels, test_size=test_size
        )
        print(f"[Pipeline] Split completato: {len(train_paths)} Train | {len(val_paths)} Validation")

        # ---------------------------------------------------------
        # 3. PREPROCESSING ONLINE (Fit sul Train, Transforms per entrambi)
        # ---------------------------------------------------------
        preprocessor = OnlinePreprocessor(self.config)
        preprocessor.fit(train_paths)

        train_transforms = preprocessor.get_transforms(is_train=True)
        val_transforms   = preprocessor.get_transforms(is_train=False)

        # ---------------------------------------------------------
        # 4. DATASETS E DATALOADERS (Lazy Loading)
        # ---------------------------------------------------------
        train_dataset = MosquitoDataset(train_paths, train_labels, transforms=train_transforms)
        val_dataset   = MosquitoDataset(val_paths,   val_labels,   transforms=val_transforms)

        batch_size  = self.config.get("batch_size", 32)
        num_workers = self.config.get("num_workers", 4)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        # ---------------------------------------------------------
        # 5. INIZIALIZZAZIONE MODELLO
        # ---------------------------------------------------------
        model = MosquitoNet(
            model_name=self.config.get("model", "resnet18"),
            pretrained=self.config.get("pretrained", True),
            num_classes=len(self.config.get("classes", [])),
            freeze_weights=self.config.get("freeze_pretrained_weights", True),
        )

        # ---------------------------------------------------------
        # 6. SETUP OTTIMIZZATORE
        # ---------------------------------------------------------
        lr = self.config.get("learning_rate", 1e-3)
        if self.config.get("freeze_pretrained_weights", True):
            optimizer = torch.optim.Adam(model.custom_head.parameters(), lr=lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # ---------------------------------------------------------
        # 7. ADDESTRAMENTO
        # ---------------------------------------------------------
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=self.device,
            config=self.config,
        )

        trainer.fit()

        # ---------------------------------------------------------
        # 8. VALUTAZIONE
        # ---------------------------------------------------------
        print("\n[Pipeline] Avvio valutazione...")

        # Config salvata qui: solo se il training è completato con successo,
        # garantendo che config.json e metriche siano sempre nella stessa cartella.
        evaluator.save_config()

        with open(trainer.history_path, "r", encoding="utf-8") as f:
            history = json.load(f)

        eval_data = np.load(trainer.eval_data_path)
        probs   = eval_data["y_probs"]
        targets = eval_data["y_true"]

        evaluator.evaluate_fold(
            fold_idx=1,
            history=history,
            probs=probs,
            targets=targets.astype(int),
        )
        evaluator.finalize()

        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETATA CON SUCCESSO ✅")
        print(f"   Output salvato in: {run_dir}")
        print("="*60)
