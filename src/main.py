"""
Entry point principale per il progetto MOSQ-AI.
Contiene la configurazione globale e avvia la Pipeline di addestramento.

Esecuzione:
    conda activate mosq-ai-env
    python src/main.py
"""

import os
import sys
from datetime import datetime

# Radice del progetto (cartella che contiene src/, data/, saved_models/, ...)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import Pipeline


def main():
    # Timestamp generato all'avvio: identifica univocamente il run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_name = "mobilenetv2"

    # =========================================================================
    # DIZIONARIO DI CONFIGURAZIONE GLOBALE (Pannello di Controllo)
    # =========================================================================
    config = {
        # --- Dati ---
        "data_path": os.path.join(PROJECT_ROOT, "data", "preprocessed"),
        "classes": ["aedes", "anopheles", "culex", "non_zanzare"],

        # Lascia vuota la lista [] per usare TUTTI i dataset, oppure
        # specificane alcuni es: ["mendeley", "bioscan"]
        "allowed_datasets": ["chula", "bioscan"],

        "val_split": 0.2,
        "seed": 42,
        "num_workers": 4,  # Abbassa a 2 o 0 se hai problemi di memoria RAM o su Windows

        # --- Preprocessing ---
        # "imagenet", "custom_z_score", o "custom_min_max"
        "norm_strategy": "imagenet",
        # True = applica augmentation sul train set, False = solo normalizzazione
        "use_augmentation": False,

        # --- Modello ---
        # Modelli supportati: resnet18, resnet50, efficientnetb0, mobilenetv2, nasnetmobile, mobilenet
        "model": model_name,
        "pretrained": True,
        "freeze_pretrained_weights": False,

        # --- Addestramento ---
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 50,
        "patience": 12,  # Early stopping

        # --- Output ---
        # La cartella del run è: saved_models/{model}_{timestamp}
        "output_dir": os.path.join(PROJECT_ROOT, "saved_models", f"{model_name}_{timestamp}"),
    }

    # =========================================================================
    # ESECUZIONE
    # =========================================================================
    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
