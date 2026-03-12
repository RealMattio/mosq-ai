"""
Entry point principale per il progetto MOSQ-AI.
Contiene la configurazione globale e avvia la Pipeline di addestramento.

Esecuzione standard (usa il modello di default):
    conda activate mosq-ai-env
    python src/main.py

Esecuzione con modello specifico:
    python src/main.py --model mobilenetv2
"""

import os
import sys
import argparse
from datetime import datetime

# Radice del progetto (cartella che contiene src/, data/, saved_models/, ...)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import Pipeline


def parse_arguments():
    """Configura l'interprete della linea di comando."""
    parser = argparse.ArgumentParser(
        description="Avvia la pipeline di addestramento MOSQ-AI."
    )
    
    # Definiamo i modelli ammessi in una lista per passarla a 'choices'
    allowed_models = [
        "resnet18", 
        "resnet50", 
        "efficientnetb0", 
        "mobilenetv2", 
        "nasnetmobile", 
        "mobilenet"
    ]
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="resnet18", # Modello di default se non specifichi nulla
        choices=allowed_models,
        help=f"Specifica l'architettura da addestrare. Modelli ammessi: {', '.join(allowed_models)}."
    )
    
    return parser.parse_args()


def main():
    # 1. Legge gli argomenti da terminale
    args = parse_arguments()
    model_name = args.model
    
    # Timestamp generato all'avvio: identifica univocamente il run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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

    print(f"Avvio addestramento per il modello: {model_name.upper()}")
    print(f"I risultati saranno salvati in: {config['output_dir']}")

    # =========================================================================
    # ESECUZIONE
    # =========================================================================
    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()