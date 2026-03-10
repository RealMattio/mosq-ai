"""
ModelEvaluator — Classe di valutazione per la pipeline MOSQ-AI.

Per ogni fold genera:
  - Grafico della training history (loss e F1 su train/val)
  - Curva ROC (one-vs-rest per ogni classe + media macro)
  - Matrice di confusione (normalizzata per riga)

Salva in JSON:
  - Metriche per classe (precision, recall, f1-score, support)
  - Medie macro e weighted tra tutte le classi
  - Accuracy globale

Alla fine (finalize) aggrega le metriche di tutti i fold
(media e deviazione standard).

Struttura output:
    saved_models/{model_name}_{timestamp}/
    ├── config.json
    ├── fold_1/
    │   ├── training_history.png
    │   ├── roc_curve.png
    │   ├── confusion_matrix.png
    │   └── metrics.json
    ├── fold_2/  (se k-fold)
    │   └── ...
    └── aggregate_metrics.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


class ModelEvaluator:
    """
    Valuta i risultati di training per ogni fold e aggrega le metriche finali.

    Args:
        config:     Dizionario di configurazione della pipeline.
        output_dir: Cartella radice in cui salvare tutti gli output del run
                    (es. saved_models/resnet18_20260310_153000/).
        classes:    Lista ordinata dei nomi delle classi
                    (es. ["aedes", "anopheles", "culex", "non_zanzare"]).
    """

    def __init__(self, config: dict[str, Any], output_dir: Path, classes: list[str]):
        self.config     = config
        self.output_dir = Path(output_dir)
        self.classes    = classes
        self.n_classes  = len(classes)
        self._fold_metrics: list[dict] = []   # accumulatore metriche per fold
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # API pubblica                                                         #
    # ------------------------------------------------------------------ #

    def evaluate_fold(
        self,
        fold_idx: int,
        history: dict[str, list[float]],
        probs: np.ndarray,
        targets: np.ndarray,
    ) -> dict:
        """
        Genera grafici e metriche per un singolo fold di addestramento.

        Args:
            fold_idx: Indice del fold (1-based).
            history:  Dict con chiavi "train_loss", "val_loss",
                      "train_acc", "val_acc", "train_f1", "val_f1".
            probs:    Array (N, n_classes) di probabilità softmax.
            targets:  Array (N,) di label vere (interi).

        Returns:
            Dict con le metriche calcolate per questo fold.
        """
        fold_dir = self.output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        preds = np.argmax(probs, axis=1)

        self._plot_training_history(fold_dir, history, fold_idx)
        self._plot_roc_curve(fold_dir, probs, targets)
        self._plot_confusion_matrix(fold_dir, preds, targets)

        metrics = self._compute_metrics(preds, targets)
        self._save_metrics(fold_dir, metrics, fold_idx)

        self._fold_metrics.append(metrics)
        print(f"[Evaluator] Fold {fold_idx} — Accuracy: {metrics['accuracy']:.4f} | "
              f"Macro F1: {metrics['macro_avg']['f1-score']:.4f}")
        return metrics

    def save_config(self) -> None:
        """Serializza il dizionario di configurazione in config.json."""
        config_path = self.output_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, default=str)
        print(f"[Evaluator] Config salvata in: {config_path}")

    def finalize(self) -> None:
        """
        Calcola e salva le metriche aggregate (media ± std) su tutti i fold.
        Chiamare dopo evaluate_fold() di tutti i fold.
        """
        if not self._fold_metrics:
            print("[Evaluator] Nessun fold da aggregare.")
            return

        aggregate = self._aggregate_metrics()
        agg_path  = self.output_dir / "aggregate_metrics.json"
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(aggregate, f, indent=2)
        print(f"[Evaluator] Metriche aggregate salvate in: {agg_path}")

    # ------------------------------------------------------------------ #
    # Grafici                                                              #
    # ------------------------------------------------------------------ #

    def _plot_training_history(
        self, fold_dir: Path, history: dict, fold_idx: int
    ) -> None:
        """Due subplot: Loss (train/val) e F1 (train/val) per epoch."""
        epochs = range(1, len(history["train_loss"]) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Training History — Fold {fold_idx} "
            f"({self.config.get('model', 'model')})",
            fontsize=13, fontweight="bold",
        )

        # — Loss —
        ax = axes[0]
        ax.plot(epochs, history["train_loss"], label="Train Loss", color="#4C72B0", lw=2)
        ax.plot(epochs, history["val_loss"],   label="Val Loss",   color="#DD8452", lw=2, linestyle="--")
        ax.set_title("Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        # — F1 Score —
        ax = axes[1]
        ax.plot(epochs, history["train_f1"], label="Train F1 (macro)", color="#4C72B0", lw=2)
        ax.plot(epochs, history["val_f1"],   label="Val F1 (macro)",   color="#DD8452", lw=2, linestyle="--")
        ax.set_title("F1 Score (macro)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        fig.savefig(fold_dir / "training_history.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_roc_curve(
        self, fold_dir: Path, probs: np.ndarray, targets: np.ndarray
    ) -> None:
        """ROC one-vs-rest per ogni classe + curva macro-average."""
        # Binarizza i target per one-vs-rest
        targets_bin = label_binarize(targets, classes=list(range(self.n_classes)))

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
                  "#8172B3", "#937860", "#DA8BC3", "#CCB974"]

        all_fpr   = np.linspace(0, 1, 300)
        mean_tprs = np.zeros_like(all_fpr)

        for i, cls_name in enumerate(self.classes):
            fpr, tpr, _ = roc_curve(targets_bin[:, i], probs[:, i])
            roc_auc     = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=1.5,
                    label=f"{cls_name} (AUC = {roc_auc:.3f})")
            mean_tprs += np.interp(all_fpr, fpr, tpr)

        # Macro average
        mean_tprs /= self.n_classes
        mean_auc   = auc(all_fpr, mean_tprs)
        ax.plot(all_fpr, mean_tprs, color="black", lw=2.5, linestyle="--",
                label=f"Macro avg (AUC = {mean_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.5)  # random baseline
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title("ROC Curve (One-vs-Rest)", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        fig.savefig(fold_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_confusion_matrix(
        self, fold_dir: Path, preds: np.ndarray, targets: np.ndarray
    ) -> None:
        """Matrice di confusione normalizzata per riga (recall per cella)."""
        cm      = confusion_matrix(targets, preds)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(self.n_classes))
        ax.set_yticks(range(self.n_classes))
        ax.set_xticklabels(self.classes, rotation=30, ha="right", fontsize=10)
        ax.set_yticklabels(self.classes, fontsize=10)
        ax.set_xlabel("Predetto", fontsize=11)
        ax.set_ylabel("Reale", fontsize=11)
        ax.set_title("Confusion Matrix (normalizzata per riga)",
                     fontsize=12, fontweight="bold")

        # Annotazioni nelle celle
        thresh = 0.5
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                color = "white" if cm_norm[i, j] > thresh else "black"
                ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                        ha="center", va="center", fontsize=9,
                        color=color, fontweight="bold")

        plt.tight_layout()
        fig.savefig(fold_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # Metriche                                                             #
    # ------------------------------------------------------------------ #

    def _compute_metrics(self, preds: np.ndarray, targets: np.ndarray) -> dict:
        """
        Calcola precision, recall, f1-score, support per classe
        e le medie macro e weighted.
        """
        report = classification_report(
            targets, preds,
            target_names=self.classes,
            output_dict=True,
            zero_division=0,
        )

        per_class = {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"],    4),
                "f1-score":  round(report[cls]["f1-score"],  4),
                "support":   int(report[cls]["support"]),
            }
            for cls in self.classes
        }

        def _round_avg(key: str) -> dict:
            return {
                "precision": round(report[key]["precision"], 4),
                "recall":    round(report[key]["recall"],    4),
                "f1-score":  round(report[key]["f1-score"],  4),
                "support":   int(report[key]["support"]),
            }

        return {
            "accuracy":     round(report["accuracy"], 4),
            "per_class":    per_class,
            "macro_avg":    _round_avg("macro avg"),
            "weighted_avg": _round_avg("weighted avg"),
        }

    def _save_metrics(self, fold_dir: Path, metrics: dict, fold_idx: int) -> None:
        payload = {"fold": fold_idx, **metrics}
        with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # ------------------------------------------------------------------ #
    # Aggregazione fold                                                    #
    # ------------------------------------------------------------------ #

    def _aggregate_metrics(self) -> dict:
        """Media e std su tutti i fold per ogni metrica."""
        n = len(self._fold_metrics)
        metric_keys = ["precision", "recall", "f1-score"]

        # Per classe
        per_class_agg: dict[str, dict] = {}
        for cls in self.classes:
            per_class_agg[cls] = {}
            for mk in metric_keys:
                vals = [fm["per_class"][cls][mk] for fm in self._fold_metrics]
                per_class_agg[cls][f"{mk}_mean"] = round(float(np.mean(vals)), 4)
                per_class_agg[cls][f"{mk}_std"]  = round(float(np.std(vals)),  4)
            supports = [fm["per_class"][cls]["support"] for fm in self._fold_metrics]
            per_class_agg[cls]["support_total"] = int(np.sum(supports))

        # Macro e weighted avg
        def _agg_avg(key: str) -> dict:
            out = {}
            for mk in metric_keys:
                vals = [fm[key][mk] for fm in self._fold_metrics]
                out[f"{mk}_mean"] = round(float(np.mean(vals)), 4)
                out[f"{mk}_std"]  = round(float(np.std(vals)),  4)
            return out

        acc_vals = [fm["accuracy"] for fm in self._fold_metrics]

        return {
            "num_folds":      n,
            "accuracy_mean":  round(float(np.mean(acc_vals)), 4),
            "accuracy_std":   round(float(np.std(acc_vals)),  4),
            "per_class":      per_class_agg,
            "macro_avg":      _agg_avg("macro_avg"),
            "weighted_avg":   _agg_avg("weighted_avg"),
        }
