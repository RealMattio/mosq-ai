"""
Riorganizza le immagini dei dataset raw in data/organized_raw/ con una
cartella per ciascuna delle quattro classi del task:

    data/organized_raw/
    ├── aedes/
    ├── anopheles/
    ├── culex/
    └── non_zanzare/

Dataset gestiti
---------------
- data/raw/chula/     (Kaggle: cyberthorn/chula-mosquito-classification)
- data/raw/masud/     (Kaggle: masud1901/mosquito-dataset-for-classification-cnn)
- data/raw/obb/       (Zenodo: OBB_dataset — tutto → non_zanzare)
- data/raw/bioscan/   (BIOSCAN-1M — già suddiviso per classe)
- data/raw/dryad/     (Dryad: Malaria vector mosquito images — già per classe)
- data/raw/mendeley/  (Mendeley: Dataset of Vector Mosquito Images — già per classe)
- data/raw/roboflow/  (Roboflow Universe — classe ricavata dal nome cartella)

Le immagini vengono COPIATE (non spostate) e rinominate con il prefisso
dataset__cartella per evitare collisioni, es.:
    chula__Ae-aegypti__img0001.jpg

Uso
---
    conda activate mosq-ai-env
    python src/data_prep/organize_data.py [--dry-run]

Opzioni
-------
    --dry-run   Mostra cosa verrebbe fatto senza copiare nulla.
"""

import argparse
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

ROOT    = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "organized_raw"

CLASSES = ("aedes", "anopheles", "culex", "non_zanzare")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Mappatura (dataset, nome_cartella) → classe  [Chula + Masud]
FOLDER_CLASS_MAP: dict[tuple[str, str], str] = {
    # --- CHULA ---
    ("chula", "Ae-aegypti"):          "aedes",
    ("chula", "Ae-albopictus"):       "aedes",
    ("chula", "Ae-vexans"):           "aedes",
    ("chula", "An-tessellatus"):      "anopheles",
    ("chula", "Cx-quinquefasciatus"): "culex",
    ("chula", "Cx-vishnui"):          "culex",
    # --- MASUD ---
    ("masud", "AEDES"):               "aedes",
    ("masud", "ANOPHELES"):           "anopheles",
    ("masud", "CULEX"):               "culex",
}

MASUD_SUBDIR = "Mosquito_dataset"

# Dataset già organizzati per classe: la sottocartella è il nome della classe
PREORGANIZED_DATASETS = ("bioscan", "dryad", "mendeley")

# Roboflow: ricava la classe dal nome della cartella progetto
def _roboflow_class(folder_name: str) -> str | None:
    n = folder_name.lower()
    if "anopheles" in n: return "anopheles"
    if "culex"     in n: return "culex"
    if "aedes" in n or "aegypti" in n or "albopictus" in n: return "aedes"
    return None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def safe_dest(dst_dir: Path, prefix: str, filename: str) -> Path:
    """Restituisce un path di destinazione unico; aggiunge _N se già esiste."""
    candidate = dst_dir / f"{prefix}__{filename}"
    if not candidate.exists():
        return candidate
    stem, suffix = Path(filename).stem, Path(filename).suffix
    n = 1
    while True:
        candidate = dst_dir / f"{prefix}__{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1

# ---------------------------------------------------------------------------
# Raccolta sorgenti
# ---------------------------------------------------------------------------

def collect_sources() -> list[tuple[Path, str]]:
    """Ritorna una lista di (path_immagine, classe_target)."""
    sources: list[tuple[Path, str]] = []

    # ── Chula e Masud (mappatura esplicita per cartella) ─────────────────
    for (dataset, folder_name), target_class in FOLDER_CLASS_MAP.items():
        if dataset == "masud":
            src_dir = RAW_DIR / dataset / MASUD_SUBDIR / folder_name
        else:
            src_dir = RAW_DIR / dataset / folder_name

        if not src_dir.is_dir():
            print(f"  [AVVISO] Cartella non trovata, saltata: {src_dir}")
            continue
        for img in src_dir.rglob("*"):
            if img.is_file() and is_image(img):
                sources.append((img, target_class))

    # ── OBB: tutto → non_zanzare ─────────────────────────────────────────
    obb_dir = RAW_DIR / "obb" / "OBB_dataset" / "images"
    if obb_dir.is_dir():
        for img in obb_dir.rglob("*"):
            if img.is_file() and is_image(img):
                sources.append((img, "non_zanzare"))
    else:
        print(f"  [AVVISO] Cartella OBB non trovata, saltata: {obb_dir}")

    # ── Dataset già organizzati per classe (bioscan, dryad, mendeley) ────
    for dataset in PREORGANIZED_DATASETS:
        dataset_dir = RAW_DIR / dataset
        if not dataset_dir.is_dir():
            print(f"  [AVVISO] Cartella {dataset} non trovata, saltata.")
            continue
        for cls_dir in dataset_dir.iterdir():
            if not cls_dir.is_dir() or cls_dir.name not in CLASSES:
                continue
            for img in cls_dir.rglob("*"):
                if img.is_file() and is_image(img):
                    sources.append((img, cls_dir.name))

    # ── Roboflow: classe ricavata dal nome cartella progetto ─────────────
    roboflow_dir = RAW_DIR / "roboflow"
    if roboflow_dir.is_dir():
        for project_dir in roboflow_dir.iterdir():
            if not project_dir.is_dir():
                continue
            target_class = _roboflow_class(project_dir.name)
            if target_class is None:
                print(f"  [AVVISO] Classe non riconosciuta per Roboflow/{project_dir.name}, saltata.")
                continue
            for img in project_dir.rglob("*"):
                if img.is_file() and is_image(img):
                    sources.append((img, target_class))
    else:
        print(f"  [AVVISO] Cartella roboflow non trovata, saltata.")

    return sources

# ---------------------------------------------------------------------------
# Copia
# ---------------------------------------------------------------------------

def copy_images(
    sources: list[tuple[Path, str]],
    dry_run: bool,
) -> dict[str, int]:
    """Copia le immagini in OUT_DIR/<classe>/. Ritorna conteggio per classe."""
    counts: dict[str, int] = {c: 0 for c in CLASSES}

    if not dry_run:
        for cls in CLASSES:
            (OUT_DIR / cls).mkdir(parents=True, exist_ok=True)

    for img_path, target_class in sources:
        rel    = img_path.relative_to(RAW_DIR)
        parts  = rel.parts
        dataset = parts[0]
        folder  = parts[-2]   # cartella immediata contenente il file
        prefix  = f"{dataset}__{folder}"

        dst_dir = OUT_DIR / target_class
        dst     = safe_dest(dst_dir, prefix, img_path.name)

        if dry_run:
            print(f"  [dry-run] {img_path.relative_to(ROOT)} → {dst.relative_to(ROOT)}")
        else:
            shutil.copy2(img_path, dst)

        counts[target_class] += 1

    return counts

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Organizza immagini raw nelle 4 classi.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostra le operazioni senza copiare file.")
    args = parser.parse_args()

    print("=" * 60)
    print("MOSQ-AI — Organizzazione dataset")
    print("=" * 60)
    print(f"Sorgente : {RAW_DIR}")
    print(f"Dest     : {OUT_DIR}")
    if args.dry_run:
        print("[MODALITÀ DRY-RUN — nessun file verrà copiato]")
    print()

    datasets = "CHULA, MASUD, OBB, BIOSCAN, DRYAD, MENDELEY, ROBOFLOW"
    print(f"Raccolta immagini da {datasets}...")
    sources = collect_sources()
    print(f"  Trovate {len(sources):,} immagini totali.\n")

    if not sources:
        print("[ERRORE] Nessuna immagine trovata. Hai eseguito gli script di download?")
        raise SystemExit(1)

    print("Copia in corso..." if not args.dry_run else "Elenco operazioni:")
    counts = copy_images(sources, dry_run=args.dry_run)

    print()
    print("=" * 60)
    print("Riepilogo" + (" (dry-run)" if args.dry_run else ""))
    print("=" * 60)
    total = 0
    for cls in CLASSES:
        print(f"  {cls:<15} {counts[cls]:>8,} immagini")
        total += counts[cls]
    print(f"  {'TOTALE':<15} {total:>8,} immagini")
    if not args.dry_run:
        print(f"\nImmagini salvate in: {OUT_DIR}")


if __name__ == "__main__":
    main()
