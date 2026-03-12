import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import time


class Trainer:
    """
    Gestisce l'addestramento, la validazione, l'Early Stopping e il calcolo delle metriche.
    Agnostico rispetto al modello: riceve il modello, i dataloader e addestra.
    """
    def __init__(
        self, 
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Dict[str, Any]
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
        self.epochs = config.get("epochs", 50)
        self.patience = config.get("patience", 10) # Per l'Early Stopping
        self.num_classes = config.get("num_classes", 4)
        
        # Output paths
        self.output_dir = Path(config.get("output_dir", "models/runs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.output_dir / "best_model.pt"
        self.history_path = self.output_dir / "training_history.json"
        self.eval_data_path = self.output_dir / "best_eval_data.npz"

        # 1. Calcolo dinamico dei pesi delle classi per sbilanciamento
        class_weights = self._compute_class_weights()
        
        # 2. Loss Function: in PyTorch nn.CrossEntropyLoss equivale alla 
        # SparseCategoricalCrossentropy e applica il softmax internamente!
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))

        # Variabili di stato per History e Early Stopping
        self.history = {
            "train_loss": [], "val_loss": [], 
            "train_acc": [], "val_acc": [], 
            "train_f1": [], "val_f1": []
        }
        self.best_val_f1 = 0.0
        self.epochs_no_improve = 0

    def _compute_class_weights(self) -> torch.Tensor:
        """
        Calcola i pesi delle classi scorrendo le label del train_loader.
        Formula: weight = totale_campioni / (num_classi * campioni_per_classe)
        Le classi rare avranno un peso > 1, quelle abbondanti < 1.
        """
        print("[Trainer] Calcolo dei Class Weights dal train_loader...")
        class_counts = np.zeros(self.num_classes)
        
        for _, targets in self.train_loader:
            unique, counts = np.unique(targets.numpy(), return_counts=True)
            class_counts[unique] += counts
            
        total_samples = np.sum(class_counts)
        # Aggiungiamo epsilon per evitare divisioni per zero
        class_weights = total_samples / (self.num_classes * (class_counts + 1e-6))
        
        print(f"[Trainer] Pesi delle classi calcolati: {np.round(class_weights, 3)}")
        return torch.FloatTensor(class_weights)

    def train_epoch(self) -> Tuple[float, float, float]:
            self.model.train()
            running_loss = 0.0
            all_preds = []
            all_targets = []

            # 1. Definiamo il formato custom (nota l'aggiunta di {postfix} alla fine)
            custom_format = "{l_bar}\033[38;5;46m{bar}\033[0m| {n_fmt}/{total_fmt} [\033[36m{elapsed}<{remaining}\033[0m, {rate_fmt}] {postfix}"
            
            # 2. Inizializziamo la barra di tqdm con il tema AI/Computer Vision
            pbar = tqdm(
                self.train_loader, 
                desc="👁️ Estr. Features & 🧠 Aggiornamento Pesi", 
                ascii=" ░▒▓█", 
                bar_format=custom_format,
                colour="green",
                ncols=115 # Leggermente più larga per far spazio alla loss
            )

            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(inputs) # Restituisce i raw logits
                
                loss = self.criterion(logits, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # 3. Aggiorniamo la barra in tempo reale con la batch loss attuale
                pbar.set_postfix({"batch_loss": f"\033[33m{loss.item():.4f}\033[0m"})

            epoch_loss = running_loss / len(self.train_loader.dataset) # type: ignore

            epoch_acc = accuracy_score(all_targets, all_preds)
            # Usiamo macro F1 perché le classi sono sbilanciate
            epoch_f1 = f1_score(all_targets, all_preds, average='macro') 
            
            return epoch_loss, epoch_acc, epoch_f1 # type: ignore


    @torch.no_grad()
    def validate_epoch(self) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = [] # Per la Curva ROC

        # 1. Formato custom: abbiamo inserito il codice ANSI \033[33m per colorare i blocchi di giallo
        custom_format = "{l_bar}\033[33m{bar}\033[0m| {n_fmt}/{total_fmt} [\033[36m{elapsed}<{remaining}\033[0m, {rate_fmt}] {postfix}"
        
        # 2. Inizializziamo la barra con un tema da "Controllo Qualità"
        pbar = tqdm(
            self.val_loader, 
            desc="🛡️ Validazione & 📊 Check Metriche", 
            ascii=" ░▒▓█", 
            bar_format=custom_format,
            colour="yellow",
            ncols=115
        )

        # Disabilitiamo il calcolo dei gradienti per risparmiare memoria e velocizzare la GPU
        with torch.no_grad():
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
                running_loss += loss.item() * inputs.size(0)

                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # 3. Aggiorniamo la barra. Uso il ciano (\033[36m) per la loss in modo che stacchi bene dal giallo
                pbar.set_postfix({"val_loss": f"\033[36m{loss.item():.4f}\033[0m"})

        epoch_loss = running_loss / len(self.val_loader.dataset) # type: ignore

        epoch_acc = accuracy_score(all_targets, all_preds)
        epoch_f1 = f1_score(all_targets, all_preds, average='macro')

        return epoch_loss, epoch_acc, epoch_f1, np.array(all_probs), np.array(all_targets) # type: ignore

    def fit(self):
        """Loop principale di addestramento con Early Stopping."""
        print(f"\n[Trainer] Inizio addestramento per massimo {self.epochs} epoche.")
        
        for epoch in range(1, self.epochs + 1):
            print("\n" + "="*30 + f" Epoch {epoch:03d}/{self.epochs} " + "="*30)
            start_time = time.time()
            train_loss, train_acc, train_f1 = self.train_epoch()
            val_loss, val_acc, val_f1, val_probs, val_targets = self.validate_epoch()
            epoch_time = time.time() - start_time

            # Aggiorna storico
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["train_f1"].append(train_f1)
            self.history["val_f1"].append(val_f1)

            print(f"Epoch {epoch:03d}/{self.epochs} | "
                  f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | "
                  f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f} | "
                  f"Time: {epoch_time:.2f}s")

            # Early Stopping Check (Basato su F1 Score di validazione)
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.epochs_no_improve = 0
                
                # Salva i pesi del modello migliore
                torch.save(self.model.state_dict(), self.best_model_path)
                
                # Salva le probabilità e i target per stampare ROC e Confusion Matrix in seguito
                np.savez(self.eval_data_path, y_probs=val_probs, y_true=val_targets)
                print(f" -> Nuovo best model salvato! (F1: {self.best_val_f1:.4f})")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"\n[Early Stopping] Nessun miglioramento per {self.patience} epoche. Addestramento fermato.")
                    break
        # Salva la history delle epoche per tracciare le curve Loss/Accuracy
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        
        print(f"\n[Trainer] Completato. Miglior F1 Validation: {self.best_val_f1:.4f}")
        print(f"[Trainer] Dati salvati in: {self.output_dir}")