"""
Download dei dataset Lab da Kaggle e Zenodo.

Dataset scaricati:
  Kaggle:
    - cyberthorn/chula-mosquito-classification          -> data/raw/chula
    - masud1901/mosquito-dataset-for-classification-cnn -> data/raw/masud
  Zenodo:
    - DOI 10.5281/zenodo.17199050 (OBB_dataset)         -> data/raw/obb

Prerequisiti:
  - Kaggle: ~/.kaggle/kaggle.json con credenziali API (permessi 600)
            Ottienilo su kaggle.com -> Settings -> API -> Create New Token
  - Zenodo: nessuna credenziale richiesta (dataset pubblico)

Uso:
    conda activate mosq-ai-env
    python src/data_prep/download_datasets.py
"""

import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

KAGGLE_DATASETS = [
    {
        "id": "cyberthorn/chula-mosquito-classification",
        "dest": "chula",
    },
    {
        "id": "masud1901/mosquito-dataset-for-classification-cnn",
        "dest": "masud",
    },
]

ZENODO_DATASETS = [
    {
        "url": "https://zenodo.org/records/17199050/files/OBB_dataset.zip?download=1",
        "dest": "obb",
        "filename": "OBB_dataset.zip",
    },
]


# ---------------------------------------------------------------------------
# Kaggle
# ---------------------------------------------------------------------------

def download_kaggle_datasets() -> None:
    try:
        from kaggle import KaggleApi
    except OSError as e:
        print(f"[ERRORE] Credenziali Kaggle non trovate: {e}")
        print("Assicurati che ~/.kaggle/kaggle.json esista e abbia permessi 600.")
        raise SystemExit(1)

    api = KaggleApi()
    api.authenticate()

    for ds in KAGGLE_DATASETS:
        dest_dir = RAW_DIR / ds["dest"]
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Kaggle | {ds['id']}] Download in corso -> {dest_dir}")
        api.dataset_download_files(ds["id"], path=str(dest_dir), quiet=False)

        zip_files = list(dest_dir.glob("*.zip"))
        if not zip_files:
            print(f"  Nessun .zip trovato in {dest_dir}, potrebbe essere già estratto.")
            continue

        zip_path = zip_files[0]
        print(f"  Estrazione di {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        zip_path.unlink()
        print(f"  Fatto. Contenuto in: {dest_dir}")


# ---------------------------------------------------------------------------
# Zenodo
# ---------------------------------------------------------------------------

def download_zenodo_datasets() -> None:
    for ds in ZENODO_DATASETS:
        dest_dir = RAW_DIR / ds["dest"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        zip_path = dest_dir / ds["filename"]

        print(f"\n[Zenodo | {ds['dest']}] Download in corso -> {dest_dir}")

        response = requests.get(ds["url"], stream=True, timeout=60)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))

        print(f"  Estrazione di {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        zip_path.unlink()
        print(f"  Fatto. Contenuto in: {dest_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    download_kaggle_datasets()
    download_zenodo_datasets()
    print("\nTutti i dataset scaricati in:", RAW_DIR)


if __name__ == "__main__":
    main()
