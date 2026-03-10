# MOSQ-AI: Smart Edge Mosquito Trap 🦟

## Obiettivo del Progetto
MOSQ-AI è un progetto di Ricerca e Sviluppo mirato alla progettazione del lato software (Computer Vision) di una trappola per zanzare intelligente. Il focus è lo sviluppo di modelli di Intelligenza Artificiale (Object Detection e Instance Segmentation) leggeri e ottimizzati per l'inferenza real-time su dispositivi edge a risorse limitate (hardware target: Raspberry Pi 5 e Raspberry Pi Zero 2 W).

Il sistema deve identificare, localizzare spazialmente (tramite Bounding Box) e segmentare le zanzare, classificandole correttamente e scartando insetti non target per evitare sprechi di energia e false attivazioni hardware.

Le classi di output previste sono:
1. `aedes` (es. Zanzara Tigre, *Aedes aegypti*, *Aedes albopictus*)
2. `culex` (Zanzara comune)
3. `anopheles`
4. `altro_insetto` (Classe di rigetto per falsi positivi: mosche, api, moscerini)

## I Task di Valutazione (Domain Shift Analysis)
La sfida ingegneristica principale del progetto è garantire che l'AI funzioni all'interno della trappola fisica, in condizioni di luce e sfondi complessi. Pertanto, l'addestramento e i test comparativi dei modelli (es. YOLOv8, custom PyTorch) verranno strutturati e misurati rigorosamente su tre domini di dati differenti:

* **Dominio 1: Laboratorio (Lab Dataset)**
    Immagini di zanzare ad alta risoluzione, isolate o fotografate in condizioni controllate con sfondi omogenei.
* **Dominio 2: Dati Sintetici (GenAI Background Replacement)**
    Immagini del Dominio 1 a cui lo sfondo originale è stato sostituito tramite modelli generativi. L'obiettivo è applicare la *Domain Randomization* per simulare artificialmente gli interni di una trappola (reti, buio, riflessi LED) e aumentare la robustezza del modello senza dover raccogliere dati manuali.
* **Dominio 3: Sul Campo (Wild Dataset)**
    Immagini catturate nel mondo reale (es. derivate dal *Mosquito Alert Challenge*), con qualità variabile, scarsa illuminazione e sfondi rumorosi. Questo rappresenta il benchmark finale per validare l'effettiva capacità di generalizzazione del modello prima del deploy su Raspberry Pi.

Tutti i modelli verranno prima addestrati su i dati provenienti da questi tre domini, successivamente verranno testati sugli altri due domini per avere un confronto incrociato tra performance del modello e applicabilità dello stesso.

## Gestione e Download dei Dataset

La cartella `data/` non è tracciata da Git. Il recupero dei dati avviene tramite script dedicati in `src/data_prep/`. La pipeline completa richiede tre passaggi in sequenza.

---

### Passaggio 1 — Dataset principali (Kaggle + Zenodo)

Scarica i dataset di base: Chula (Kaggle), Masud (Kaggle) e OBB (Zenodo).

```bash
conda activate mosq-ai-env
python src/data_prep/download_datasets.py
```

**Prerequisiti Kaggle:** creare il file `~/.kaggle/kaggle.json` con le proprie credenziali.
1. Accedere a [kaggle.com](https://www.kaggle.com) → *Settings* → *API* → *Create New Token*
2. Salvare il file scaricato in `~/.kaggle/kaggle.json`
3. Impostare i permessi corretti: `chmod 600 ~/.kaggle/kaggle.json`

**Zenodo:** nessuna credenziale richiesta.

---

### Passaggio 2 — Dataset aggiuntivi per il bilanciamento delle classi

Scarica dataset supplementari per le classi minoritarie (`anopheles`, `culex`, `non_zanzare`).

```bash
conda activate mosq-ai-env
python src/data_prep/download_additional_datasets.py --roboflow-key <API_KEY>
```

**Prerequisiti Roboflow:** ottenere una API key gratuita.
1. Registrarsi su [roboflow.com](https://roboflow.com)
2. Andare su *Settings* → *API* e copiare la chiave personale
3. Passarla allo script con il flag `--roboflow-key`

**Mendeley (download manuale):** il dataset [Dataset of Vector Mosquito Images](https://data.mendeley.com/datasets/88s6fvgg2p/4) richiede autenticazione per il download programmatico.
1. Accedere alla pagina del dataset su Mendeley Data
2. Cliccare *Download All* per scaricare il file `Dataset of Vector Mosquito Images.zip`
3. Salvare lo ZIP in `data/raw/`

**Dryad (download manuale):** il dataset [Malaria vector mosquito images](https://datadryad.org/dataset/doi:10.5061/dryad.z08kprr92) è soggetto a rate limiting severo sull'API pubblica.
1. Accedere alla pagina del dataset su Dryad
2. Cliccare *Download Dataset* per scaricare lo ZIP completo
3. Salvare lo ZIP in `data/raw/`

Una volta scaricati gli ZIP di Mendeley e Dryad, estrarli nelle cartelle organizzate per classe:

```bash
conda activate mosq-ai-env
python src/data_prep/download_additional_datasets.py --skip-mendeley  # solo Roboflow
# Gli ZIP vengono estratti automaticamente se presenti in data/raw/
```

> **Nota:** lo script `download_additional_datasets.py` estrae automaticamente gli ZIP di Mendeley e Dryad se trovati in `data/raw/` con i nomi attesi:
> - `data/raw/Dataset of Vector Mosquito Images.zip`
> - `data/raw/Malaria_vector_mosquito_images_2_dryad.zip`

---

### Passaggio 3 — Download BIOSCAN-1M (bilanciamento `non_zanzare`)

Scarica immagini di insetti non-zanzara dal dataset [BIOSCAN-1M](https://zenodo.org/records/8030065) tramite range requests HTTP sul file ZIP remoto (non viene scaricato l'intero archivio da 7 GB).

```bash
conda activate mosq-ai-env
python src/data_prep/download_bioscan.py [--workers 6] [--seed 42]
```

Nessuna credenziale richiesta. Il download riprende automaticamente da dove si è interrotto. I file intermedi vengono messi in cache in `data/raw/bioscan/`:
- `metadata.tsv` (1.2 GB, scaricato una sola volta)
- `zip_cd_index.pkl` (indice del file ZIP remoto, 148 MB, scaricato una sola volta)

---

### Passaggio 4 — Organizzazione in `data/organized_raw/`

Dopo aver completato i download, organizzare tutte le immagini nelle 4 cartelle di classe:

```bash
conda activate mosq-ai-env
python src/data_prep/organize_data.py
```

---

### Dataset disponibili

| Nome | Sorgente | Cartella locale | Classi | Immagini |
|---|---|---|---|---|
| Chula Mosquito Classification | [Kaggle](https://www.kaggle.com/datasets/cyberthorn/chula-mosquito-classification) | `data/raw/chula/` | aedes, anopheles, culex | ~52.000 |
| Mosquito Dataset CNN | [Kaggle](https://www.kaggle.com/datasets/masud1901/mosquito-dataset-for-classification-cnn) | `data/raw/masud/` | aedes, anopheles, culex | 3.000 |
| OBB Mosquito Detection | [Zenodo](https://doi.org/10.5281/zenodo.17199050) | `data/raw/obb/` | non_zanzare | 802 |
| Dataset of Vector Mosquito Images | [Mendeley](https://data.mendeley.com/datasets/88s6fvgg2p/4) | `data/raw/mendeley/` | aedes, anopheles, culex | 2.684 |
| Malaria vector mosquito images | [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.z08kprr92) | `data/raw/dryad/` | anopheles, culex | 969 |
| BIOSCAN-1M (campionamento) | [Zenodo](https://zenodo.org/records/8030065) | `data/raw/bioscan/` | non_zanzare, culex | ~27.400 |
| Culex datasets (4×) | [Roboflow Universe](https://universe.roboflow.com) | `data/raw/roboflow/` | culex | 2.034 |
| Anopheles dataset | [Roboflow Universe](https://universe.roboflow.com/data-j491o/anopheles-3) | `data/raw/roboflow/` | anopheles | 4.505 |

**Distribuzione per classe dopo organizzazione** (dati `data/raw/`, pre-deduplicazione):

| Classe | Immagini disponibili |
|---|---|
| `aedes` | ~29.000 |
| `anopheles` | ~14.000 |
| `culex` | ~22.000 |
| `non_zanzare` | ~20.000 |

**Note sul mapping delle classi:**
- `aedes`: *Ae. aegypti*, *Ae. albopictus*, *Ae. vexans*
- `anopheles`: *An. tessellatus*, *An. stephensi*, *An. funestus*
- `culex`: *Cx. quinquefasciatus*, *Cx. vishnui*
- `non_zanzare`: insetti non-Culicidae da OBB dataset e BIOSCAN-1M

**Nota sul dataset OBB:** le immagini contengono esclusivamente insetti non-zanzara e vengono usate come classe di rigetto `non_zanzare`. Le label OBB incluse nel dataset originale non vengono utilizzate in questa pipeline.

## Organizzazione della Repository
La struttura adotta un approccio modulare per separare nettamente i dati, la prototipazione, il codice di pipeline e gli artefatti generati.

```
.
├── data/              # [NON COMMITTATA] Dataset grezzi, organizzati e preprocessati.
│   ├── raw/           # Immagini originali divise per sorgente (chula, masud, obb, bioscan, …)
│   ├── organized_raw/ # Immagini copiate e rinominate nelle 4 cartelle di classe
│   └── preprocessed/  # Output di preprocess_organized_data.py (224×224, CLAHE, letterbox)
├── models/            # Pesi esportati post-training (.pt, .onnx, .tflite)
├── notebooks/         # Jupyter Notebooks per EDA, prototipazione e analisi
├── README.md
└── src/               # Core logico del progetto — una cartella per fase di pipeline
    ├── dataset_org/   # Download, organizzazione e labeling dei dati grezzi
    │   ├── download_datasets.py           # Kaggle (Chula, Masud) + Zenodo (OBB)
    │   ├── download_additional_datasets.py # Roboflow, Mendeley, Dryad
    │   ├── download_bioscan.py            # Campionamento BIOSCAN-1M (non_zanzare)
    │   └── organize_data.py              # Copia in organized_raw/ per classe
    ├── data_loading/  # DataLoader PyTorch, split train/val/test, sampler
    ├── preprocessing/ # Preprocessing offline delle immagini
    │   └── preprocess_organized_data.py  # CLAHE + letterboxing + resize 224×224
    ├── models/        # Definizione architetture, caricamento pesi, export ONNX/TFLite
    ├── training/      # Training loop, ottimizzatori, scheduler, checkpoint
    └── evaluation/    # Metriche, grafici, domain shift analysis
        └── analyze_organized_raw_data.py # Distribuzione immagini per classe/dataset
```