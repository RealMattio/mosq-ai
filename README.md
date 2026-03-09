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
Per mantenere la repository leggera, nessun dato grezzo o immagine è tracciato su Git (la cartella `data/` è ignorata).
Il recupero dei dati è gestito in modo programmatico tramite lo script `src/data_prep/download_datasets.py`, che si connette alle API pubbliche (Kaggle, Roboflow, Zenodo), scarica gli archivi e li estrae nella directory `data/raw/`.

Per popolare localmente i dataset prima di avviare qualsiasi pipeline di addestramento:

```bash
conda activate mosq-ai-env
python src/data_prep/download_datasets.py
```

> **Prerequisiti:**
> - **Kaggle:** è necessario avere un file `~/.kaggle/kaggle.json` valido con le proprie credenziali API (ottenibile da kaggle.com → Settings → API → Create New Token).
> - **Zenodo:** nessuna credenziale richiesta, i dataset sono pubblici.

### Dataset disponibili *(sezione in aggiornamento)*

> Nuovi dataset verranno integrati progressivamente. La tabella seguente riflette lo stato attuale.

#### Dominio 1 — Laboratorio (Lab Dataset)

| Nome | Sorgente | Cartella locale | Classi | Immagini totali |
|---|---|---|---|---|
| Chula Mosquito Classification | [Kaggle](https://www.kaggle.com/datasets/cyberthorn/chula-mosquito-classification) | `data/raw/chula/` | Ae-aegypti, Ae-albopictus, Ae-vexans, An-tessellatus, Cx-quinquefasciatus, Cx-vishnui, Misc | ~61.400 |
| Mosquito Dataset for Classification (CNN) | [Kaggle](https://www.kaggle.com/datasets/masud1901/mosquito-dataset-for-classification-cnn) | `data/raw/masud/` | AEDES, ANOPHELES, CULEX | 3.000 |
| OBB Mosquito Detection Dataset | [Zenodo](https://doi.org/10.5281/zenodo.17199050) | `data/raw/obb/OBB_dataset/` | zanzare (classe singola) | 802 (682 train / 120 val) |

**Nota sul mapping delle classi:** i dataset Lab usano nomenclature a granularità variabile. Prima del training verrà applicato un mapping verso le 4 classi di progetto: `aedes` (Ae-aegypti, Ae-albopictus, Ae-vexans), `culex` (Cx-*), `anopheles` (An-*), `altro_insetto` (Misc e classi non target).

**Nota sul dataset OBB:** le etichette sono in formato OBB (Oriented Bounding Box) con 4 vertici per annotazione — adatto a modelli che supportano bounding box ruotati (es. YOLOv8-OBB). DOI: [10.5281/zenodo.17199050](https://doi.org/10.5281/zenodo.17199050).

## Organizzazione della Repository
La struttura adotta un approccio modulare per separare nettamente i dati, la prototipazione, il codice di pipeline e gli artefatti generati.

```
.
├── data/        # [NON COMMITTATA] File system locale per dataset (Lab, GenAI, Wild) divisi in train/val/test.
├── models/      # Salvataggio dei pesi esportati post-training (.pt, .onnx, .tflite).
├── notebooks/   # Jupyter Notebooks per EDA (Exploratory Data Analysis), test di inferenza e prototipazione script generativi.
├── README.md    # Documentazione di progetto.
└── src/         # Core logico del progetto Python.
    ├── data_prep/      # Script per il download, la standardizzazione e l'augmentation/background replacement.
    ├── training/       # Logica di addestramento PyTorch, configurazione iperparametri e validazione.
    └── edge_inference/ # Codice ottimizzato per il deployment (OpenCV, inferenza ONNX/TFLite per Raspberry).
```