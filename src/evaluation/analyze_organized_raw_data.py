"""
Analisi statistica della distribuzione dei dati in data/organized_raw/.

Produce un grafico a barre impilate (stacked bar chart) che mostra
per ogni classe il numero di immagini suddivise per dataset di origine.
Il grafico viene salvato in src/evaluation/graphs/.

Uso
---
    conda activate mosq-ai-env
    python src/evaluation/analyze_organized_raw_data.py
    
    # Per escludere uno o più dataset:
    python src/evaluation/analyze_organized_raw_data.py --exclude bioscan masud
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# ---------------------------------------------------------------------------
# Percorsi
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT / "data" / "organized_raw"
GRAPHS_DIR  = Path(__file__).resolve().parent / "graphs"
OUTPUT_FILE = GRAPHS_DIR / "class_distribution.png"

CLASSES = ["aedes", "anopheles", "culex", "non_zanzare"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Colori per dataset (palette accessibile)
DATASET_COLORS = {
    "chula":    "#4C72B0",
    "masud":    "#DD8452",
    "obb":      "#55A868",
    "bioscan":  "#C44E52",
    "dryad":    "#8172B3",
    "mendeley": "#937860",
    "roboflow": "#DA8BC3",
}

# ---------------------------------------------------------------------------
# Raccolta dati
# ---------------------------------------------------------------------------

def collect_counts(excluded_datasets: set[str] | None = None) -> pd.DataFrame:
    """
    Scansiona organized_raw/ e conta le immagini per (classe, dataset).
    I filename hanno il formato: dataset__folder__originalname.ext
    
    Args:
        excluded_datasets: Set di nomi di dataset (in minuscolo) da ignorare.
    """
    if excluded_datasets is None:
        excluded_datasets = set()
        
    counts: dict[tuple[str, str], int] = defaultdict(int)

    for cls in CLASSES:
        cls_dir = DATA_DIR / cls
        if not cls_dir.is_dir():
            print(f"  [AVVISO] Cartella non trovata: {cls_dir}")
            continue

        for img in cls_dir.iterdir():
            if not img.is_file() or img.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
                
            # Estrai il dataset dal prefisso del nome file
            dataset = img.name.split("__")[0].lower()
            
            # Salta se il dataset è nella lista degli esclusi
            if dataset in excluded_datasets:
                continue
                
            counts[(cls, dataset)] += 1

    # Se non ci sono dati dopo l'esclusione, ritorna un DataFrame vuoto
    if not counts:
        return pd.DataFrame()

    # Costruisci DataFrame con indice (classe) e colonne (dataset)
    datasets = sorted({ds for (_, ds) in counts})
    data = {
        ds: [counts.get((cls, ds), 0) for cls in CLASSES]
        for ds in datasets
    }
    df = pd.DataFrame(data, index=CLASSES)
    return df


# ---------------------------------------------------------------------------
# Grafico Migliorato
# ---------------------------------------------------------------------------

def plot_distribution(df: pd.DataFrame, excluded_str: str = "") -> None:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    totals_per_class = df.sum(axis=1)
    grand_total = int(totals_per_class.sum())

    fig, ax = plt.subplots(figsize=(11, 6.5))

    bar_width = 0.55
    bottom = pd.Series([0] * len(CLASSES), index=CLASSES, dtype=float)

    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

    for dataset in df.columns:
        values = df[dataset]
        color  = DATASET_COLORS.get(dataset.lower(), "#999999") 
        bars   = ax.bar(
            CLASSES,
            values,
            bar_width,
            bottom=bottom,
            label=dataset.upper(),
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3 
        )
        
        for bar, val in zip(bars, values):
            total_bar_height = totals_per_class[CLASSES[list(bars).index(bar)]]
            # Gestisci il caso di barre con altezza 0 per evitare divisioni per zero
            if total_bar_height > 0 and val > (total_bar_height * 0.04): 
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(val):,}",
                    ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold",
                )
        bottom += values

    for i, (cls, total) in enumerate(totals_per_class.items()):
        # Mostra il totale solo se è maggiore di zero
        if total > 0:
            ax.text(
                i,
                total + (totals_per_class.max() * 0.02),
                f"{int(total):,}",
                ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="#333333",
            )

    title = f"Distribuzione Immagini per Classe — Dataset 'organized_raw'\nTotale Complessivo: {grand_total:,} immagini"
    if excluded_str:
        title += f"\n(Esclusi: {excluded_str})"

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20, color="#222222")
    
    ax.set_xlabel("CLASSE", fontsize=10, fontweight="bold", color="#555555", labelpad=10)
    ax.set_ylabel("NUMERO DI IMMAGINI", fontsize=10, fontweight="bold", color="#555555", labelpad=10)
    
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    
    # Gestisci il caso in cui max sia 0
    y_max = totals_per_class.max()
    if y_max > 0:
        ax.set_ylim(0, y_max * 1.15)
    
    ax.legend(
        title="Dataset Sorgente", 
        fontsize=10, 
        title_fontsize=11, 
        loc="upper left", 
        bbox_to_anchor=(1.02, 1), 
        frameon=False 
    )
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    
    ax.tick_params(axis="x", labelsize=12, bottom=False) 
    ax.tick_params(axis="y", labelsize=10, length=0)     

    plt.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight") 
    plt.close(fig)
    print(f"Grafico migliorato salvato in: {OUTPUT_FILE}")


# ---------------------------------------------------------------------------
# Stampa riepilogo testuale
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, excluded_str: str = "") -> None:
    totals_per_class = df.sum(axis=1)
    grand_total = int(totals_per_class.sum())

    print("\n" + "=" * 52)
    print("Distribuzione immagini — organized_raw")
    if excluded_str:
         print(f"(Escludendo: {excluded_str})")
    print("=" * 52)

    # Intestazione
    datasets = list(df.columns)
    header = f"{'Classe':<14}" + "".join(f"{ds.upper():>10}" for ds in datasets) + f"{'TOTALE':>10}"
    print(header)
    print("-" * len(header))

    for cls in CLASSES:
        row = f"{cls:<14}"
        for ds in datasets:
            row += f"{df.loc[cls, ds]:>10,}"
        row += f"{int(totals_per_class[cls]):>10,}"
        print(row)

    print("-" * len(header))
    footer = f"{'TOTALE':<14}"
    for ds in datasets:
        footer += f"{int(df[ds].sum()):>10,}"
    footer += f"{grand_total:>10,}"
    print(footer)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analizza la distribuzione delle immagini nei dataset.")
    parser.add_argument(
        "--exclude", 
        nargs="+", 
        default=[], 
        help="Lista di dataset da escludere (es. --exclude bioscan masud)"
    )
    args = parser.parse_args()

    excluded_set = {ds.lower() for ds in args.exclude}
    excluded_str = ", ".join(sorted(excluded_set))

    print(f"Scansione di: {DATA_DIR}")
    if excluded_set:
         print(f"Escludendo i dataset: {excluded_str}")

    df = collect_counts(excluded_set)

    if df.empty or df.sum().sum() == 0:
        print("[ERRORE] Nessuna immagine trovata (o tutte escluse). Hai eseguito organize_data.py?")
        raise SystemExit(1)

    print_summary(df, excluded_str)
    plot_distribution(df, excluded_str)


if __name__ == "__main__":
    main()