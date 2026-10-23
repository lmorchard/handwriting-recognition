# Project Handoff: Handwriting Recognition (Phase 1 Wrap-up & Phase 2 Launch)

**Date**: 2026-10-22  
**Status**: **Phase 1 Complete (Digital E-Ink Archive Transcribed & Operational)**  
**Code Repository (Public)**: `git@github.com:lmorchard/handwriting-recognition.git` (`main` at `51deb7a`)  
**Data Repository (Private)**: `ssh://git@myriad-docker:22222/lmorchard/handwriting-data.git` (`main` at `17afe6f`)  
**Production Model**: `handwriting-data/models/best_model` (2,721 lines, **2.27% CER**, loss `0.2400`)  
**Vault Reference Page**: `[[agent/pages/Handwriting OCR Project Idea]]`

---

## 1. What Was Accomplished in Phase 1

1. **Full Archive Transcription**:
   - Transcribed **522 out of 523 journal volumes** (43,199 lines of handwriting spanning 2014–2026) in 310 minutes on Portia's Apple Silicon M5 Max.
   - 99.8% success rate. Exactly 1 file failed (`2026-03-10-summary.pdf`) due to a truncated PDF stream.
   - Transcripts live in the Obsidian vault under `notepads/transcriptions/{YYYY}/{YYYY-MM-DD}.md`.
   - Raw baseline snapshot preserved in `notepads/transcriptions_backup_raw_20261022.tar.gz` (644 KB).
2. **Bounded Vault Lexicon Cleanup (`clean_transcript.py`)**:
   - Applied a hard Levenshtein clamp ($\le 2$) anchored to 10 years of personal Obsidian notes (166k words).
   - Applied **3,806 word corrections** across 509 files without hallucinating or smoothing authentic phrasing (*"awhile frettingly"*, *Catsby*, *Cheddars*, *Dan Hon*, *Lamy*, *AL-star*, *BOOX*).
3. **Zero-Friction Active Learning Flywheel (`harvest_corrections.py`)**:
   - Eliminated the labeling UI bottleneck for daily use.
   - The user edits Markdown directly in Obsidian. `harvest_corrections.py` diffs against `<stem>.raw.md` (or the backup tarball), crops the exact physical ink stroke strips from the source PDF, and updates `spot_corrections/dataset.csv`.
   - **179 verified lines harvested** into the active simmer buffer across `2026-01-01`, `2026-10-21`, and `2026-10-22`.
4. **Asymmetric Homelab Split**:
   - **Linux Ingest VM (Proxmox CPU-only)**: Always-on daily ingestion and harvesting engine.
   - **Portia (MacBook Pro M5 Max GPU)**: On-demand training forge for multi-epoch training and 90-second simmer bursts.
5. **Clean Public/Private Architecture**:
   - Decoupled code (`handwriting-recognition` on GitHub) from private datasets (`handwriting-data` on Forgejo `myriad-docker:22222`).
   - Squashed 29 development commits (`7132087`) to permanently expunge personal journal text from public GitHub history.
6. **Zero-Argument Automation (`Makefile`)**:
   - `make transcribe`: Automatically finds untranscribed PDFs in `notepads/journals/`, transcribes, polishes, and places `.md` and `.raw.md` directly into the vault.
   - `make harvest`: Automatically finds edited notes, extracts ink strips, appends to the dataset, commits, and pushes to Forgejo.

---

## 2. Directory & Machine Layout

```text
Host: Portia (~/devel/mine/) & Linux VM (workspace/)
├── handwriting-recognition/         [Public GitHub Repo: 7132087..51deb7a]
│   ├── Makefile                    # Zero-arg commands (transcribe, harvest, simmer, etc.)
│   ├── .env                        # Local paths (ignored by Git)
│   ├── extract_lines.py            # PDF vector ink extraction & landscape split
│   ├── train_trocr.py              # Fine-tuning harness (MPS & CUDA)
│   ├── train_simmer.py             # Fast GPU burst with experience replay
│   ├── transcribe_journal.py       # Full production batch transcriber
│   ├── clean_transcript.py         # Levenshtein vault lexicon post-processor
│   ├── harvest_corrections.py      # Obsidian diff ink stroke harvester
│   ├── label_ui.py                 # Multi-batch local web review tool
│   ├── FINDINGS.md                 # 18-section technical report
│   ├── BLOG_POST_OUTLINE.md        # 7-Act narrative arc for blog.lmorchard.com
│   ├── docs/
│   │   ├── ARCHITECTURE.md         # Compute split & repo firewall
│   │   └── PDF_EXTRACTION.md       # Vector /Ink reverse engineering
│   └── charts/                     # 300-DPI publication plots
│
└── handwriting-data/               [Private Forgejo Repo: 0xDECAFBAD hub]
    ├── batches/                    # 30 verified training batches (2,721 lines)
    ├── spot_corrections/           # 179 harvested lines + cropped PNG strips
    ├── vocab/                      # 166k vault unigrams & English dictionary
    └── models/                     # [Ignored by Git]
        └── best_model/             # 1.3 GB fine-tuned weights (2.27% CER)
```

---

## 3. Daily Operational Cheat Sheet

For daily maintenance, the user runs two commands:

```bash
# 1. Morning / After writing on BOOX tablet:
make transcribe
# -> Finds new PDF, transcribes with TrOCR, applies vault polish, drops .md and .raw.md in vault

# 2. Human touch:
# -> Open Obsidian, read note, fix any typos directly in Markdown

# 3. After editing in Obsidian:
make harvest
# -> Auto-diffs against .raw.md, crops ink strips, updates dataset, commits & pushes to Forgejo
```

For historical archive entries:
* **Re-transcribe with latest model**: `make transcribe PDF=2023-06-13.pdf` $\to$ edit $\to$ `make harvest NOTE=...`
* **Fix typos in existing archive note**: Edit in Obsidian $\to$ `make harvest NOTE=notepads/transcriptions/2024/2024-05-12.md`

---

## 4. Phase 2 Scope & Roadmap

When picking up this project for the next phase, focus on these three tracks:

### Track 1: First Simmer Retraining Run
* The active simmer buffer in `handwriting-data/spot_corrections/dataset.csv` has reached **179 lines** (well past the 50–100 line threshold).
* **Command to run on Portia**:
  ```bash
  make simmer
  ```
* **Verify**:
  * Ensure experience replay mixes the 179 new spot lines with random samples from the 30 baseline batches (prevents catastrophic forgetting).
  * Run a fast 90-second GPU burst (~3 epochs at lr=2e-5).
  * Evaluate CER on validation lines and update `best_model`.

### Track 2: Draft the Blog Post for `blog.lmorchard.com`
* Use `BLOG_POST_OUTLINE.md` (which is already structured into a 7-Act narrative tuned to Les Orchard's writing voice).
* Focus on:
  * The "Tuna vs. Turkey" moment (why text LLMs commit literary forgery on personal journals).
  * The vector PDF discovery (Onyx putting dot grids in raster and handwriting in vector `/Ink`).
  * The steep scaling curve (84 lines $\to$ 16% CER $\to$ 2,721 lines $\to$ 2.27% CER).
  * The homelab asymmetric split and zero-friction active learning.
* Include the high-res charts in `charts/` (`cer_by_epoch.png`, `scaling_law.png`).

### Track 3: Physical Paper Notebook Scans (The True Computer Vision Challenge)
* **The Problem**: Digital e-ink was "easy mode" because Onyx gave us vector stroke coordinates without lighting, perspective, or scan noise.
* **Requirements for Physical Notebooks**:
  1. **Dot-Grid Removal on Raster**:
     - Unlike BOOX PDFs, physical notebooks have real ink drawn on printed dot-grid paper.
     - Implement morphological opening / adaptive thresholding or a small U-Net / perceptual filter to strip the 5mm dot grid without thinning handwriting strokes.
  2. **Perspective & Page Dewarping**:
     - Handle phone photos / flatbed scans: desk angle correction, boundary corner detection, and page curvature unwarping.
  3. **Line Segmentation on Real Paper**:
     - Because there are no vector `/Annots`, use CRAFT, YOLOv8-obb, or horizontal projection profile slicing on the cleaned raster.
  4. **Fine-Tuning on Paper Ingest**:
     - TrOCR already knows the author's small-caps glyphs from Phase 1. The fine-tuning task in Phase 2 is adapting to ballpoint / fountain pen stroke thickness and paper fiber bleed.

---

## 5. Key Invariants for Future Maintainers

* **Never use unconstrained generative LLMs for transcription cleanup**: Text-only LLMs will erase authentic voice (*"awhile frettingly"*, pet names, homelab jargon). All post-processing must use the Levenshtein-clamped lexicon in `clean_transcript.py` ($\le 2$ edit distance).
* **Never commit personal datasets to the public repo**: All batches, CSVs, and image strips belong strictly in `../handwriting-data` (Forgejo). `handwriting-recognition/.gitignore` is strictly configured to enforce this.
* **Keep `models/` out of Git**: Model checkpoints (`model.safetensors` ~1.3 GB) live in `handwriting-data/models/` and are ignored by Git to prevent repo bloating.
