"""
Download campionato da BIOSCAN-1M per bilanciare le classi minoritarie.

Strategia:
  1. Legge i conteggi correnti in data/organized_raw/ e calcola i target.
  2. Scarica il metadata TSV da Zenodo (1.2 GB, cached su disco) e lo
     parsa con pandas caricando solo le colonne utili.
  3. Mostra quante immagini sono disponibili per classe nel dataset e
     calcola i target effettivi (±15% di flessibilità).
  4. Scarica il Central Directory del ZIP tramite HTTP range request
     (148 MB, cached su disco).
  5. Per ogni immagine selezionata fa un range request mirato (~8 KB).
  6. Scrive manifest.csv per la riproducibilità.

File scaricati e messi in cache in data/raw/bioscan/:
  - metadata.tsv          (1.2 GB, indice tassonomico)
  - zip_cd_index.pkl      (148 MB parsato → dict pickle)

Uso:
    conda activate mosq-ai-env
    python src/data_prep/download_bioscan.py [--target N] [--seed 42] [--workers 4]
"""

import argparse
import csv
import pickle
import random
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

ROOT          = Path(__file__).resolve().parents[2]
ORGANIZED_DIR = ROOT / "data" / "organized_raw"
OUT_DIR       = ROOT / "data" / "raw" / "bioscan"
METADATA_PATH = OUT_DIR / "metadata.tsv"
INDEX_CACHE   = OUT_DIR / "zip_cd_index.pkl"
MANIFEST_PATH = OUT_DIR / "manifest.csv"

CLASSES   = ("aedes", "anopheles", "culex", "non_zanzare")
IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
REQUEST_TIMEOUT = 30

METADATA_URL = (
    "https://zenodo.org/api/records/8030065/files/"
    "BIOSCAN_Insect_Dataset_metadata.tsv/content"
)
ZIP_URL   = "https://zenodo.org/api/records/8030065/files/cropped_256.zip/content"
CD_OFFSET = 7_016_178_802
CD_SIZE   = 155_034_082

# Colonne TSV da caricare (le altre vengono ignorate → memoria ridotta)
TSV_COLS  = ["image_file", "genus", "family", "order"]

# ---------------------------------------------------------------------------
# Sessione HTTP robusta
# ---------------------------------------------------------------------------

def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.5,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET", "HEAD"])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://",  adapter)
    return s

SESSION = make_session()

# ---------------------------------------------------------------------------
# Conteggi correnti
# ---------------------------------------------------------------------------

def read_current_counts() -> dict[str, int]:
    counts = {}
    for cls in CLASSES:
        d = ORGANIZED_DIR / cls
        counts[cls] = sum(
            1 for p in d.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ) if d.is_dir() else 0
    return counts

# ---------------------------------------------------------------------------
# Fase 1: download metadata TSV (cached) + selezione image_file
# ---------------------------------------------------------------------------

def download_metadata() -> None:
    """Scarica il TSV su disco se non già presente."""
    if METADATA_PATH.exists():
        print(f"  Metadata già in cache: {METADATA_PATH} "
              f"({METADATA_PATH.stat().st_size / 1e9:.2f} GB)")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Download metadata TSV → {METADATA_PATH}")
    r = SESSION.get(METADATA_URL, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    bar = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
               desc="  metadata.tsv")
    with open(METADATA_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            bar.update(len(chunk))
    bar.close()


def classify_df(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge colonna 'target_class' al DataFrame."""
    genus  = df["genus"].str.strip().str.lower()
    family = df["family"].str.strip().str.lower()
    order  = df["order"].str.strip().str.lower()

    conditions = [
        genus == "anopheles",
        genus == "culex",
        genus == "aedes",
        (family != "culicidae") & (~family.isin(["", "not_classified"])) & (order != ""),
    ]
    choices = ["anopheles", "culex", "_skip_aedes", "non_zanzare"]

    df = df.copy()
    df["target_class"] = pd.Categorical(
        pd.Series(pd.NA, index=df.index, dtype=object)
    )
    result = np.select(conditions, choices, default=pd.NA) # type: ignore
    df["target_class"] = result
    return df


def select_image_files(
    targets: dict[str, int],
    seed: int,
) -> dict[str, list[str]]:
    """
    Carica il TSV, classifica le righe e campiona i file necessari.
    Mostra la disponibilità reale per classe prima di campionare.
    """
    print(f"\n  Carico metadata TSV con pandas (colonne: {TSV_COLS})…")
    df = pd.read_csv(
        METADATA_PATH,
        sep="\t",
        usecols=lambda c: c.strip().lower() in TSV_COLS, # type: ignore
        dtype=str,
        low_memory=False,
    )
    # Normalizza nomi colonne
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.dropna(subset=["image_file"])
    df = df[df["image_file"].str.strip() != ""]

    print(f"  Righe totali nel TSV: {len(df):,}")

    df = classify_df(df)
    df = df[~df["target_class"].isin(["_skip_aedes"]) & df["target_class"].notna()]

    print("\n  Disponibilità nel dataset BIOSCAN-1M:")
    available: dict[str, int] = {}
    for cls in ("anopheles", "culex", "non_zanzare"):
        n = int((df["target_class"] == cls).sum())
        available[cls] = n
        print(f"    {cls:<14}: {n:>8,} immagini disponibili  "
              f"(target: {targets.get(cls, 0):>7,})")

    rng = random.Random(seed)
    selected: dict[str, list[str]] = {}
    for cls, target_n in targets.items():
        pool = df[df["target_class"] == cls]["image_file"].str.strip().tolist()
        rng.shuffle(pool)
        actual = min(target_n, len(pool))
        selected[cls] = pool[:actual]
        delta = actual - target_n
        note  = f" (↓{abs(delta):,} sotto target)" if delta < 0 else ""
        print(f"    {cls:<14}: {actual:>8,} selezionate{note}")

    return selected

# ---------------------------------------------------------------------------
# Fase 2: Central Directory ZIP64 (cached)
# ---------------------------------------------------------------------------

def _parse_zip64_extra(extra: bytes, need_off: bool, need_comp: bool) -> tuple[int, int]:
    lo = cs = 0
    i = 0
    while i + 4 <= len(extra):
        tag, size = struct.unpack_from("<HH", extra, i)
        data = extra[i + 4: i + 4 + size]
        if tag == 0x0001:
            vals = [struct.unpack_from("<Q", data, p)[0]
                    for p in range(0, len(data) - 7, 8)]
            if need_comp:
                cs = vals[1] if len(vals) >= 2 else (vals[0] if vals else 0)
            if need_off:
                lo = vals[2] if len(vals) >= 3 else (vals[-1] if vals else 0)
        i += 4 + size
    return lo, cs


def build_zip_index(needed: set[str]) -> dict[str, tuple[int, int, int]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if INDEX_CACHE.exists():
        print(f"\n  Indice ZIP già in cache: {INDEX_CACHE.name}")
        with open(INDEX_CACHE, "rb") as f:
            full_index: dict[str, tuple[int, int, int]] = pickle.load(f)
        print(f"  {len(full_index):,} voci nell'indice.")
    else:
        print(f"\n  Download Central Directory ZIP (148 MB)…")
        cd_end = CD_OFFSET + CD_SIZE - 1
        r = SESSION.get(ZIP_URL, headers={"Range": f"bytes={CD_OFFSET}-{cd_end}"},
                        stream=True, timeout=120)
        r.raise_for_status()
        bar = tqdm(total=CD_SIZE, unit="B", unit_scale=True, unit_divisor=1024,
                   desc="  Central Directory")
        cd = bytearray()
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            bar.update(len(chunk))
            cd.extend(chunk)
        bar.close()

        print("  Parsing Central Directory…")
        full_index = {}
        pos, MAX32 = 0, 0xFFFF_FFFF
        SIG = b"PK\x01\x02"
        while pos + 46 <= len(cd):
            if cd[pos:pos + 4] != SIG:
                nxt = cd.find(SIG, pos + 1)
                if nxt == -1:
                    break
                pos = nxt
                continue
            comp_method           = struct.unpack_from("<H",   cd, pos + 10)[0]
            comp_size_            = struct.unpack_from("<I",   cd, pos + 20)[0]
            fname_len, extra_len, comment_len = struct.unpack_from("<HHH", cd, pos + 28)
            local_off_            = struct.unpack_from("<I",   cd, pos + 42)[0]
            fname = cd[pos + 46: pos + 46 + fname_len].decode("utf-8", errors="replace")
            extra = cd[pos + 46 + fname_len: pos + 46 + fname_len + extra_len]

            need_o = local_off_ == MAX32
            need_c = comp_size_ == MAX32
            if need_o or need_c:
                lo, cs  = _parse_zip64_extra(extra, need_o, need_c)
                local_off  = lo if need_o else local_off_
                comp_size  = cs if need_c else comp_size_
            else:
                local_off, comp_size = local_off_, comp_size_

            base = Path(fname).name
            if base and not fname.endswith("/"):
                full_index[base] = (local_off, comp_size, comp_method)

            pos += 46 + fname_len + extra_len + comment_len

        print(f"  {len(full_index):,} file indicizzati. Salvo cache…")
        with open(INDEX_CACHE, "wb") as f:
            pickle.dump(full_index, f)

    missing = needed - set(full_index)
    if missing:
        print(f"  [AVVISO] {len(missing):,} file non trovati nell'indice ZIP.")
    return {fn: full_index[fn] for fn in needed if fn in full_index}

# ---------------------------------------------------------------------------
# Fase 3: download immagini
# ---------------------------------------------------------------------------

def _download_one(
    session: requests.Session,
    image_file: str,
    local_off: int,
    comp_size: int,
    comp_method: int,
    dest: Path,
) -> bool:
    try:
        # Header locale (30 byte)
        r1 = session.get(ZIP_URL,
                         headers={"Range": f"bytes={local_off}-{local_off + 29}"},
                         timeout=REQUEST_TIMEOUT)
        if r1.status_code not in (200, 206):
            return False
        lh = r1.content
        if len(lh) < 30 or lh[:4] != b"PK\x03\x04":
            return False
        fname_len, extra_len = struct.unpack_from("<HH", lh, 26)
        data_start = local_off + 30 + fname_len + extra_len

        # Dati compressi
        r2 = session.get(ZIP_URL,
                         headers={"Range": f"bytes={data_start}-{data_start + comp_size - 1}"},
                         timeout=REQUEST_TIMEOUT)
        if r2.status_code not in (200, 206):
            return False

        raw = (r2.content if comp_method == 0
               else zlib.decompress(r2.content, -15) if comp_method == 8
               else None)
        if raw is None:
            return False

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        return dest.stat().st_size > 500
    except Exception:
        return False


def download_class(
    cls: str,
    image_files: list[str],
    zip_index: dict[str, tuple[int, int, int]],
    workers: int,
    manifest_rows: list[dict],
) -> int:
    out_dir = OUT_DIR / cls
    out_dir.mkdir(parents=True, exist_ok=True)
    session = make_session()
    downloaded = 0

    def _task(fn: str) -> tuple[str, str]:
        dest = out_dir / fn
        if dest.exists() and dest.stat().st_size > 500:
            return fn, "already_present"
        entry = zip_index.get(fn)
        if entry is None:
            return fn, "not_in_zip"
        ok = _download_one(session, fn, *entry, dest)
        return fn, "ok" if ok else "download_failed"

    bar = tqdm(total=len(image_files), desc=f"  {cls}", unit="img")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fn, status in (f.result() for f in as_completed(
                ex.submit(_task, fn) for fn in image_files)):
            manifest_rows.append({"image_file": fn, "class": cls, "status": status})
            if status in ("ok", "already_present"):
                downloaded += 1
            bar.update(1)
    bar.close()
    session.close()
    return downloaded

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scarica immagini da BIOSCAN-1M per bilanciare le classi."
    )
    parser.add_argument("--target",  type=int, default=None,
                        help="Immagini target per classe (default: max classe attuale)")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostra disponibilità e target senza scaricare immagini")
    args = parser.parse_args()

    print("=" * 62)
    print("MOSQ-AI — Download BIOSCAN-1M")
    print("=" * 62)

    current = read_current_counts()
    target_val = args.target or max(current.values())
    targets = {
        cls: max(0, target_val - current.get(cls, 0))
        for cls in ("anopheles", "culex", "non_zanzare")
    }

    print("\nConteggio attuale (organized_raw):")
    for cls in CLASSES:
        print(f"  {cls:<14} {current.get(cls, 0):>8,}")
    print(f"\nTarget per classe: {target_val:,}")

    if not any(targets.values()):
        print("\nDataset già bilanciato, nessun download necessario.")
        return

    # Fase 1: metadata
    print("\n[1/3] Metadata TSV")
    download_metadata()
    selected = select_image_files(targets, args.seed)
    all_files = {fn for files in selected.values() for fn in files}

    print(f"\n  Totale immagini da scaricare: {len(all_files):,}")

    if args.dry_run:
        print("\n[dry-run] Nessun file scaricato.")
        return

    if not all_files:
        print("[ERRORE] Nessun file selezionato.")
        raise SystemExit(1)

    # Fase 2: indice ZIP
    print("\n[2/3] Indice ZIP Central Directory")
    zip_index = build_zip_index(all_files)

    # Fase 3: download
    print(f"\n[3/3] Download immagini ({args.workers} thread paralleli)")
    manifest_rows: list[dict] = []
    for cls, files in selected.items():
        if not files:
            print(f"  {cls}: nessun file da scaricare.")
            continue
        n = download_class(cls, files, zip_index, args.workers, manifest_rows)
        print(f"  {cls}: {n}/{len(files)} completate")

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=["image_file", "class", "status"]) \
            .writeheader()
        csv.DictWriter(f, fieldnames=["image_file", "class", "status"]) \
            .writerows(manifest_rows)

    ok   = sum(1 for r in manifest_rows if r["status"] == "ok")
    skip = sum(1 for r in manifest_rows if r["status"] == "already_present")
    fail = sum(1 for r in manifest_rows if r["status"] != "ok"
               and r["status"] != "already_present")

    print("\n" + "=" * 62)
    print(f"  Scaricate nuove   : {ok:>7,}")
    print(f"  Già presenti      : {skip:>7,}")
    print(f"  Fallite/mancanti  : {fail:>7,}")
    print(f"\n  Manifest: {MANIFEST_PATH}")
    print("  Passo successivo: aggiorna organize_data.py per includere")
    print("  data/raw/bioscan/ in organized_raw/")


if __name__ == "__main__":
    main()
