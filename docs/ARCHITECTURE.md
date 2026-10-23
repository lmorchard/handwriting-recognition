# Homelab Architecture & Repository Separation

This project separates operational roles across machines and keeps private personal data strictly isolated from public open-source code.

---

## 1. Asymmetric Homelab Split

The workload is split between two distinct machines based on hardware strengths and availability:

```
┌─────────────────────────────────────────────────────────────┐
│                    Linux Ingest VM (Proxmox)                │
│  - Always-on headless environment                           │
│  - Runs CPU-only TrOCR daily inference (~60-90s per volume) │
│  - Bounded Levenshtein vault lexicon cleanup                │
│  - Monitors Obsidian vault for incoming daily PDFs          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                Git Sync via Private Forgejo
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                    Portia (MacBook Pro M5 Max)              │
│  - Transient workstation (moves, sleeps, docks)             │
│  - On-demand training forge using Apple Silicon MPS         │
│  - Multi-epoch model fine-tuning (10 epochs in ~80 min)     │
│  - Fast spot-correction simmer bursts (~90 seconds)         │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Public Code vs. Private Data Separation

To allow open-source publication without exposing private personal journal prose or training crops:

| Repository | Host / Visibility | Contents |
| :--- | :--- | :--- |
| **`handwriting-recognition`** | **Public GitHub** (`lmorchard/handwriting-recognition`) | Modular Python scripts, Makefile, test scaffolding, documentation, and publication charts. Contains zero personal prose. |
| **`handwriting-data`** | **Private Forgejo** (`0xDECAFBAD hub`) | 30 verified training batches (2,721 lines), active-learning spot corrections buffer, vault vocabulary caches, and local fine-tuned model checkpoints (`models/best_model/`). |

Both Portia and the Linux VM maintain local checkouts of `handwriting-data` side-by-side with the code repository.

---

## 3. Configuration via `.env`

All paths and runtime options are loaded from `.env` via `config.py` with zero external dependencies.

```bash
# Directory where private training batches and spot corrections live
HANDWRITING_DATA_DIR=../handwriting-data

# Path to the fine-tuned TrOCR model checkpoint directory
HANDWRITING_MODEL_DIR=../handwriting-data/models/best_model

# Path to your Obsidian vault root directory
OBSIDIAN_VAULT_DIR=/Users/lmorchard/Documents/obsidian/main

# Optional: Path to raw baseline tarball for historical archive diffs
HANDWRITING_BACKUP_TARBALL=/Users/lmorchard/Documents/obsidian/main/notepads/transcriptions_backup_raw_20261022.tar.gz

# Local port for the labeling web UI (make label)
PORT=8080
```
