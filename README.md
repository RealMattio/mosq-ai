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
Il recupero dei dati è gestito in modo programmatico. All'interno della repository verranno creati script Python appositi per:
- Connettersi alle API pubbliche (es. Kaggle, Roboflow, Zenodo).
- Scaricare gli archivi dei vari domini (Lab e Wild).
- Normalizzare le etichette (formato YOLO) e strutturare fisicamente i file in train/val/test.

Gli script di download e preparazione andranno eseguiti per popolare localmente la directory `data/` prima di avviare qualsiasi pipeline di addestramento.

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