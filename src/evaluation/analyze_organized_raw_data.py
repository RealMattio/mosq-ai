"""
Analisi statistica della distribuzione dei dati in data/organized_raw/.

Produce un grafico a barre impilate (stacked bar chart) che mostra
per ogni classe il numero di immagini suddivise per dataset di origine.
Il grafico viene salvato in src/evaluation/graphs/.

Uso
---
    conda activate mosq-ai-env
    python src/evaluation/analyze_organized_raw_data.py
"""

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
    "chula": "#4C72B0",
    "masud": "#DD8452",
    "obb":   "#55A868",
}

# ---------------------------------------------------------------------------
# Raccolta dati
# ---------------------------------------------------------------------------

def collect_counts() -> pd.DataFrame:
    """
    Scansiona organized_raw/ e conta le immagini per (classe, dataset).
    I filename hanno il formato: dataset__folder__originalname.ext
    """
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
            dataset = img.name.split("__")[0]
            counts[(cls, dataset)] += 1

    # Costruisci DataFrame con indice (classe) e colonne (dataset)
    datasets = sorted({ds for (_, ds) in counts})
    data = {
        ds: [counts.get((cls, ds), 0) for cls in CLASSES]
        for ds in datasets
    }
    df = pd.DataFrame(data, index=CLASSES)
    return df


# ---------------------------------------------------------------------------
# Grafico
# ---------------------------------------------------------------------------

def plot_distribution(df: pd.DataFrame) -> None:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    totals_per_class = df.sum(axis=1)
    grand_total = int(totals_per_class.sum())

    fig, ax = plt.subplots(figsize=(10, 6))

    # Barre impilate
    bar_width = 0.5
    bottom = pd.Series([0] * len(CLASSES), index=CLASSES, dtype=float)

    for dataset in df.columns:
        values = df[dataset]
        color  = DATASET_COLORS.get(dataset, "#999999")
        bars   = ax.bar(
            CLASSES,
            values,
            bar_width,
            bottom=bottom,
            label=dataset.upper(),
            color=color,
            edgecolor="white",
            linewidth=0.6,
        )
        # Etichetta del segmento (solo se abbastanza alto da essere leggibile)
        for bar, val in zip(bars, values):
            if val > 150:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:,}",
                    ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold",
                )
        bottom += values

    # Totale per classe sopra ogni barra
    for cls, total in totals_per_class.items():
        ax.text(
            CLASSES.index(cls),
            total + grand_total * 0.005,
            f"{int(total):,}",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#222222",
        )

    # Decorazioni
    ax.set_title(
        f"Distribuzione immagini per classe — organized_raw\n"
        f"Totale: {grand_total:,} immagini",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Classe", fontsize=11)
    ax.set_ylabel("Numero di immagini", fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, totals_per_class.max() * 1.12)
    ax.legend(title="Dataset", fontsize=10, title_fontsize=10, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafico salvato in: {OUTPUT_FILE}")


# ---------------------------------------------------------------------------
# Stampa riepilogo testuale
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    totals_per_class = df.sum(axis=1)
    grand_total = int(totals_per_class.sum())

    print("\n" + "=" * 52)
    print("Distribuzione immagini — organized_raw")
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
    print(f"Scansione di: {DATA_DIR}")
    df = collect_counts()

    if df.empty or df.sum().sum() == 0:
        print("[ERRORE] Nessuna immagine trovata. Hai eseguito organize_data.py?")
        raise SystemExit(1)

    print_summary(df)
    plot_distribution(df)


if __name__ == "__main__":
    main()
