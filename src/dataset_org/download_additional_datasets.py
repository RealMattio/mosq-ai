"""
Download dei dataset aggiuntivi per bilanciare le classi minoritarie.

Sorgenti:
  Dryad   doi:10.5061/dryad.z08kprr92  — API pubblica, no auth (740 img)
  Mendeley 88s6fvgg2p v4               — scraping pagina pubblica
  Roboflow (5 dataset Culex, 1 Anopheles) — richiede --roboflow-key

Le immagini vengono salvate in:
  data/raw/dryad/     {aedes|anopheles|culex}/filename.jpg
  data/raw/mendeley/  {aedes|anopheles|culex}/filename.jpg
  data/raw/roboflow/  {progetto}/{split}/filename.jpg

Uso:
    conda activate mosq-ai-env
    python src/data_prep/download_additional_datasets.py
    python src/data_prep/download_additional_datasets.py --roboflow-key YOUR_KEY

Opzioni:
    --roboflow-key KEY   API key Roboflow (gratuita su roboflow.com)
    --workers N          Thread per download immagini (default: 6)
    --skip-dryad         Salta Dryad
    --skip-mendeley      Salta Mendeley
    --dry-run            Elenca i file senza scaricarli
"""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Percorsi
# ---------------------------------------------------------------------------

ROOT     = Path(__file__).resolve().parents[2]
RAW_DIR  = ROOT / "data" / "raw"
ORG_DIR  = ROOT / "data" / "organized_raw"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ---------------------------------------------------------------------------
# Sessione HTTP robusta
# ---------------------------------------------------------------------------

def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.0,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET", "HEAD"])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    s.mount("https://", adapter)
    s.mount("http://",  adapter)
    return s

SESSION = make_session()

# ---------------------------------------------------------------------------
# Helper: specie da filename / cartella
# ---------------------------------------------------------------------------

def species_from_name(name: str) -> str | None:
    """Estrae la classe (aedes/anopheles/culex) dal nome file o cartella."""
    n = name.lower()
    if "anopheles" in n:
        return "anopheles"
    if "culex" in n:
        return "culex"
    if "aedes" in n or "aegypti" in n or "albopictus" in n:
        return "aedes"
    return None


def download_file(url: str, dest: Path, session: requests.Session,
                  timeout: int = 60) -> bool:
    """Scarica un file su disco. Ritorna True se successo."""
    if dest.exists() and dest.stat().st_size > 500:
        return True
    try:
        r = session.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
        return dest.stat().st_size > 500
    except Exception:
        return False

# ---------------------------------------------------------------------------
# DRYAD  doi:10.5061/dryad.z08kprr92
# ---------------------------------------------------------------------------

DRYAD_BASE       = "https://datadryad.org"
DRYAD_VERSION_ID = 91679    # version endpoint estratto dall'API

def dryad_list_files() -> list[dict]:
    """Recupera la lista completa dei file via API paginata."""
    files = []
    url = f"{DRYAD_BASE}/api/v2/versions/{DRYAD_VERSION_ID}/files"
    while url:
        r = SESSION.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        files.extend(data.get("_embedded", {}).get("stash:files", []))
        url_next = data.get("_links", {}).get("next", {}).get("href")
        url = DRYAD_BASE + url_next if url_next else None
    return files


def download_dryad(out_dir: Path, workers: int, dry_run: bool) -> dict[str, int]:
    # Dryad impone un rate limit IP-based: usa 1 worker + delay tra richieste
    import time
    DRYAD_DELAY = 2.5   # secondi tra un download e il successivo

    print("\n── DRYAD (doi:10.5061/dryad.z08kprr92) ──────────────────")
    print("  Recupero lista file…")
    try:
        files = dryad_list_files()
    except Exception as e:
        print(f"  [ERRORE] Impossibile accedere all'API Dryad: {e}")
        return {}

    print(f"  {len(files)} file trovati nel dataset.")

    tasks: list[tuple[str, Path, str]] = []
    for f in files:
        fname   = f.get("path", "").split("/")[-1]
        file_id = f.get("id")
        mime    = f.get("mimeType", "")
        if not fname or "image" not in mime:
            continue
        cls = species_from_name(fname)
        if cls is None:
            continue
        url  = f"{DRYAD_BASE}/api/v2/files/{file_id}/download"
        dest = out_dir / cls / fname
        tasks.append((url, dest, cls))

    counts: dict[str, int] = {}
    for _, _, cls in tasks:
        counts[cls] = counts.get(cls, 0) + 1
    print(f"  Immagini per specie: " +
          ", ".join(f"{k}={v}" for k, v in counts.items()))
    eta_min = len(tasks) * DRYAD_DELAY / 60
    print(f"  Download sequenziale con delay {DRYAD_DELAY}s "
          f"(~{eta_min:.0f} min stimati)")
    if dry_run:
        return counts

    session = make_session()
    ok = 0
    bar = tqdm(total=len(tasks), desc="  Dryad", unit="img")
    for url, dest, cls in tasks:
        # Skip se già presente
        if dest.exists() and dest.stat().st_size > 500:
            ok += 1
            bar.update(1)
            continue
        # Retry manuale su 429 con backoff crescente
        wait = DRYAD_DELAY
        for attempt in range(6):
            time.sleep(wait)
            try:
                r = session.get(url, stream=True, timeout=60,
                                allow_redirects=True)
                if r.status_code == 429:
                    wait = min(wait * 2, 120)
                    tqdm.write(f"  [429] rate limit, aspetto {wait:.0f}s…")
                    continue
                r.raise_for_status()
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as f_out:
                    for chunk in r.iter_content(chunk_size=65536):
                        f_out.write(chunk)
                if dest.stat().st_size > 500:
                    ok += 1
                break
            except Exception as e:
                tqdm.write(f"  [ERR] {dest.name}: {e}")
                break
        bar.update(1)
    bar.close()
    print(f"  Scaricate: {ok}/{len(tasks)}")
    return counts

# ---------------------------------------------------------------------------
# MENDELEY  88s6fvgg2p v4
# ---------------------------------------------------------------------------

MENDELEY_DATASET_URL = "https://data.mendeley.com/datasets/88s6fvgg2p/4"

def mendeley_get_file_links() -> list[dict]:
    """
    Recupera i link di download dalla pagina HTML pubblica di Mendeley Data.
    La pagina incorpora i metadati del dataset come JSON nel tag <script>.
    """
    r = SESSION.get(MENDELEY_DATASET_URL, timeout=30,
                    headers={"Accept": "text/html"})
    r.raise_for_status()
    html = r.text

    # Mendeley incorpora i dati dataset come __NEXT_DATA__ JSON nello <script>
    match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                      html, re.DOTALL)
    if not match:
        return []

    data = json.loads(match.group(1))

    # Naviga nella struttura JSON per trovare i file
    def find_files(obj, depth=0):
        if depth > 10:
            return []
        if isinstance(obj, list):
            results = []
            for item in obj:
                results.extend(find_files(item, depth + 1))
            return results
        if isinstance(obj, dict):
            # Cerca oggetti che sembrano file con download_url
            if "download_url" in obj and "filename" in obj:
                return [obj]
            results = []
            for v in obj.values():
                results.extend(find_files(v, depth + 1))
            return results
        return []

    return find_files(data)


def download_mendeley(out_dir: Path, workers: int, dry_run: bool) -> dict[str, int]:
    print("\n── MENDELEY (88s6fvgg2p v4) ────────────────────────────")
    print("  Recupero file dalla pagina HTML pubblica…")

    try:
        files = mendeley_get_file_links()
    except Exception as e:
        print(f"  [ERRORE] Impossibile accedere alla pagina Mendeley: {e}")
        files = []

    if not files:
        print("  [AVVISO] Nessun file trovato via scraping HTML.")
        print("  Mendeley Data richiede autenticazione per il download programmatico.")
        print("  Scarica manualmente da:")
        print(f"  {MENDELEY_DATASET_URL}")
        print("  e salva le cartelle in data/raw/mendeley/{aedes,anopheles,culex}/")
        return {}

    tasks: list[tuple[str, Path, str]] = []
    for f in files:
        fname = f.get("filename", "")
        url   = f.get("download_url", "")
        if not fname or not url:
            continue
        # Determina la specie dal percorso/nome
        content_type = f.get("content_type", "")
        folder       = f.get("folder", fname)
        cls = species_from_name(folder) or species_from_name(fname)
        if cls is None:
            continue
        dest = out_dir / cls / fname
        tasks.append((url, dest, cls))

    counts: dict[str, int] = {}
    for _, _, cls in tasks:
        counts[cls] = counts.get(cls, 0) + 1

    print(f"  Immagini per specie: " +
          ", ".join(f"{k}={v}" for k, v in counts.items()))
    if dry_run or not tasks:
        return counts

    print(f"  Download con {workers} thread…")
    session = make_session()
    ok = 0
    bar = tqdm(total=len(tasks), desc="  Mendeley", unit="img")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(download_file, url, dest, session): cls
                for url, dest, cls in tasks}
        for fut in as_completed(futs):
            if fut.result():
                ok += 1
            bar.update(1)
    bar.close()
    print(f"  Scaricate: {ok}/{len(tasks)}")
    return counts

# ---------------------------------------------------------------------------
# ROBOFLOW  (5 dataset Culex + 1 Anopheles)
# ---------------------------------------------------------------------------

ROBOFLOW_DATASETS = [
    # (workspace, project, versione_default, classe_nota)
    ("techworkspace", "culex-mnj4z",              1, "culex"),
    ("gui-zi4ls",     "culex_vishnui2",            1, "culex"),
    ("gui-zi4ls",     "culex_quinquefasciatus2",   1, "culex"),
    ("gui-zi4ls",     "culex_quinquefasciatus",    1, "culex"),
    ("data-j491o",    "anopheles-3",               1, "anopheles"),
]

ROBOFLOW_API = "https://api.roboflow.com"


ROBOFLOW_FORMATS = ["yolov8-obb", "yolov8", "coco", "voc", "multiclass"]

def roboflow_get_export_url(workspace: str, project: str,
                            version: int, api_key: str) -> str | None:
    """
    Recupera l'URL di download ZIP per un dataset Roboflow.
    Prova in ordine i formati disponibili; usa la versione più alta se v1 manca.
    """
    # Controlla versioni disponibili
    proj_r = SESSION.get(f"{ROBOFLOW_API}/{workspace}/{project}",
                         params={"api_key": api_key}, timeout=30)
    if proj_r.status_code != 200:
        return None
    latest_v = proj_r.json().get("project", {}).get("versions", version)

    # Ottieni info versione per sapere i formati già esportati
    ver_r = SESSION.get(f"{ROBOFLOW_API}/{workspace}/{project}/{latest_v}",
                        params={"api_key": api_key}, timeout=30)
    if ver_r.status_code != 200:
        return None
    exports_ready = ver_r.json().get("version", {}).get("exports", [])

    # Prima prova i formati già pronti, poi gli altri
    ordered = exports_ready + [f for f in ROBOFLOW_FORMATS if f not in exports_ready]

    for fmt in ordered:
        r = SESSION.get(f"{ROBOFLOW_API}/{workspace}/{project}/{latest_v}/{fmt}",
                        params={"api_key": api_key}, timeout=30)
        if r.status_code == 200:
            link = r.json().get("export", {}).get("link")
            if link:
                return link
    return None


def download_roboflow_project(workspace: str, project: str, version: int,
                              forced_cls: str, out_dir: Path,
                              api_key: str, dry_run: bool) -> int:
    print(f"  [{forced_cls}] {workspace}/{project} v{version}…")
    export_url = roboflow_get_export_url(workspace, project, version, api_key)
    if not export_url:
        print(f"    [ERRORE] Impossibile ottenere URL export per {project}")
        return 0

    zip_path = out_dir / f"{project}.zip"
    if not dry_run:
        ok = download_file(export_url, zip_path, SESSION, timeout=300)
        if not ok:
            print(f"    [ERRORE] Download fallito per {project}")
            return 0

        # Estrai solo le immagini dal ZIP (ignora label .txt e yaml)
        import zipfile
        extract_dir = out_dir / project
        with zipfile.ZipFile(zip_path, "r") as zf:
            imgs = [n for n in zf.namelist()
                    if Path(n).suffix.lower() in IMG_EXTS
                    and "label" not in n.lower()]
            for img_name in imgs:
                # Appiattisci la struttura: salva solo il basename
                dest = extract_dir / Path(img_name).name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(img_name))
        zip_path.unlink()

        count = sum(1 for p in extract_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMG_EXTS)
        print(f"    {count} immagini estratte in {extract_dir.relative_to(ROOT)}")
        return count
    return 0


def download_roboflow(out_dir: Path, api_key: str,
                      dry_run: bool) -> dict[str, int]:
    print("\n── ROBOFLOW ────────────────────────────────────────────")
    counts: dict[str, int] = {}

    for workspace, project, version, forced_cls in ROBOFLOW_DATASETS:
        n = download_roboflow_project(workspace, project, version, forced_cls,
                                      out_dir / "roboflow", api_key, dry_run)
        counts[forced_cls] = counts.get(forced_cls, 0) + n

    return counts

# ---------------------------------------------------------------------------
# Riepilogo finale
# ---------------------------------------------------------------------------

def count_images_recursive(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.rglob("*")
               if p.is_file() and p.suffix.lower() in IMG_EXTS)


CLASSES = ("aedes", "anopheles", "culex", "non_zanzare")


def print_summary() -> None:
    print("\n" + "=" * 62)
    print("RIEPILOGO IMMAGINI PER CLASSE")
    print("=" * 62)

    sources = {
        "organized_raw (tot)": ORG_DIR,
        "raw/chula":           RAW_DIR / "chula",
        "raw/masud":           RAW_DIR / "masud",
        "raw/obb":             RAW_DIR / "obb",
        "raw/bioscan":         RAW_DIR / "bioscan",
        "raw/dryad":           RAW_DIR / "dryad",
        "raw/mendeley":        RAW_DIR / "mendeley",
        "raw/roboflow":        RAW_DIR / "roboflow",
    }

    # organized_raw: conta per classe
    print("\norganized_raw — distribuzione per classe:")
    org_total = 0
    for cls in CLASSES:
        n = count_images_recursive(ORG_DIR / cls)
        org_total += n
        print(f"  {cls:<14} {n:>8,}")
    print(f"  {'TOTALE':<14} {org_total:>8,}")

    # raw: conta per sorgente (totale, senza distinzione classe)
    print("\nraw — immagini per sorgente:")
    for label, path in list(sources.items())[1:]:
        n = count_images_recursive(path)
        if n > 0:
            print(f"  {label:<22} {n:>8,}")

    # raw/dryad e raw/mendeley: per classe se presente
    for src_name in ("dryad", "mendeley", "roboflow"):
        src_dir = RAW_DIR / src_name
        if not src_dir.is_dir():
            continue
        has_class_dirs = any((src_dir / cls).is_dir() for cls in CLASSES)
        if has_class_dirs:
            print(f"\n  {src_name} — per classe:")
            for cls in ("aedes", "anopheles", "culex"):
                n = count_images_recursive(src_dir / cls)
                if n:
                    print(f"    {cls:<14} {n:>8,}")

    print()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scarica dataset aggiuntivi per MOSQ-AI."
    )
    parser.add_argument("--roboflow-key", default=None,
                        help="API key Roboflow (gratuita su roboflow.com)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--skip-mendeley", action="store_true")
    parser.add_argument("--dry-run",       action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("MOSQ-AI — Download dataset aggiuntivi")
    print("=" * 62)

    if not args.skip_mendeley:
        download_mendeley(
            out_dir  = RAW_DIR / "mendeley",
            workers  = args.workers,
            dry_run  = args.dry_run,
        )

    if args.roboflow_key:
        download_roboflow(
            out_dir  = RAW_DIR,
            api_key  = args.roboflow_key,
            dry_run  = args.dry_run,
        )
    else:
        print("\n── ROBOFLOW ─────────────────────────────────────────────")
        print("  Saltato: fornisci --roboflow-key per scaricare i dataset Roboflow.")
        print("  API key gratuita su: https://roboflow.com  → Settings → API")
        print("  Dataset disponibili:")
        for ws, proj, ver, cls in ROBOFLOW_DATASETS:
            print(f"    [{cls}] {ws}/{proj} v{ver}")

    print_summary()


if __name__ == "__main__":
    main()
