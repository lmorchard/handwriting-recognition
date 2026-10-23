# Personal Handwriting Recognition with TrOCR

A personalized handwriting recognition pipeline to transcribe handwritten journals (Onyx BOOX e-ink tablet PDFs and physical paper notebook scans) into plain Markdown text.

Fine-tuned on **2,721 personal handwriting lines** across 5 years of journals (2022–2026), achieving **2.27% Character Error Rate (CER)** on Apple Silicon M5 Max.

---

## Key Discovery: BOOX Vector Ink Decoupling

BOOX tablets export page templates (like dot grids) as a background raster image (`/XObject`), while exporting all handwritten stylus strokes as native vector PDF `/Ink` annotations (`/Annots`).

* **Zero Dot-Grid Noise**: Handwriting can be rendered directly from vector stroke paths onto a pure white canvas—eliminating dot grids, scanning artifacts, and paper texture without computer vision morphology.
* **Geometric Line Segmentation**: Every stroke has exact bounding coordinates (`[x1, y1, x2, y2]`). Lines are grouped geometrically by vertical clustering without Tesseract, CRAFT, or OpenCV contour slicing.

---

## The Daily Workflow: Transcribe, Edit & Ingest

The daily lifecycle is designed around a zero-friction active learning flywheel:

```
[ BOOX Stylus Ink ] ──► [ Daily PDF ]
                             │
                             ▼ (1. Transcribe: ~15s on Mac / ~60s on Linux)
                     [ Raw Markdown ]
                             │
                             ▼ (2. Vault Polish: Levenshtein ≤ 2 clamp)
                     [ Polished Markdown + .raw.md baseline ]
                             │
                             ▼ (3. Human Review in Obsidian)
                     [ User-Edited Markdown ]
                             │
                             ▼ (4. Harvest Corrections)
                     [ spot_corrections/ (Physical Ink Strips + Ground Truth) ]
                             │
                             ▼ (5. Simmer Retrain: 90s GPU burst with replay)
                     [ Updated TrOCR Checkpoint ]
```

### Step 1: Transcribe & Clean Directly into Obsidian Vault

When you write an entry on your BOOX tablet and it syncs to `notepads/journals/`, just run:

```bash
make transcribe
```

**Zero Arguments Needed:** It automatically scans `notepads/journals/*.pdf`, skips anything already transcribed, and only processes new entries. *(You can still specify a single file with `make transcribe PDF=2026-10-22.pdf` if desired).*

**What this does automatically in one pass**:
1. **TrOCR Inference**: Runs line segmentation and vision recognition (~15s on M5 Max GPU, ~60–90s on Linux CPU).
2. **Lexicon Polish**: Automatically applies the Levenshtein clamp ($\le 2$) against your 166k-word vault lexicon to resolve OCR typos without generative hallucination.
3. **Direct Vault Placement**: Moves the polished Markdown straight into your Obsidian vault under `notepads/transcriptions/YYYY/YYYY-MM-DD.md`.
4. **Safety Baseline**: Leaves a `<filename>.raw.md` baseline alongside the note so diffing is ready for harvesting.

---

### Step 2: Review and Edit in Obsidian

Open Obsidian and read through `notepads/transcriptions/2026/2026-10-22.md`.
* Fix any transcription typos directly in Markdown just like editing a normal note.
* Write naturally—keep slang, pet names, or personal phrasing.

---

### Step 3: Harvest, Commit & Push in One Step

Once you finish editing in Obsidian, just run:

```bash
make harvest
```

**Zero Arguments Needed:** It automatically scans `notepads/transcriptions/` for any `.raw.md` baselines, detects which notes you edited, crops the ink lines, updates the dataset, and commits & pushes to Forgejo! *(You can still specify a single note with `make harvest NOTE=...` if desired).*

**What happens automatically**:
1. Diffs your edited Markdown against the `.raw.md` baseline.
2. Aligns word corrections to physical lines using `difflib.SequenceMatcher`.
3. Opens the source PDF, crops the exact physical ink stroke strips for those lines, and saves them to `../handwriting-data/spot_corrections/images/`.
4. Appends the image paths and your corrected text to `../handwriting-data/spot_corrections/dataset.csv`.
5. Cleans up the temporary `.raw.md` baseline file now that corrections are banked.
6. **Auto Git Commit & Push**: Automatically commits the new spot corrections and pushes to your private Forgejo repository.
7. **Simmer Alert**: Reminds you when total spot corrections reach $\ge 50$ or $100$ lines so you know when to run `make simmer`.

---

### Step 4: Version in Private Forgejo & Simmer Retrain

Commit the harvested training pairs to your private Git repository:

```bash
cd ../handwriting-data
git add spot_corrections/
git commit -m "Harvest corrections from 2026-10-22"
git push origin main
```

Whenever ~50–100 spot corrections accumulate in the simmer buffer:

```bash
cd ../handwriting-recognition
make simmer
```

This launches a **90-second GPU simmer burst** on Apple Silicon M5 Max with an experience replay buffer, baking your recent corrections into the model without catastrophic forgetting.

---

## Repository & Homelab Architecture

This project strictly separates **public code** from **private data**:

* **`handwriting-recognition` (Public GitHub)**: Open-source tooling, CLI scripts, documentation, and publication charts.
* **`handwriting-data` (Private Forgejo / `0xDECAFBAD hub`)**: 30 verified training batches (2,721 lines), active-learning spot corrections buffer, vault vocabulary caches, and local fine-tuned model checkpoints (`models/best_model/`).

### Configuration via `.env`

Copy `.env.example` to `.env` to configure your local paths:

```bash
HANDWRITING_DATA_DIR=../handwriting-data
HANDWRITING_MODEL_DIR=../handwriting-data/models/best_model
OBSIDIAN_VAULT_DIR=/Users/lmorchard/Documents/obsidian/main
PORT=8080
```

---

## Command Reference

| Target | Description |
| :--- | :--- |
| `make transcribe PDF=...` | Transcribe a single BOOX PDF to Markdown |
| `make clean-transcript FILE=...` | Apply Levenshtein vault lexicon cleanup to a note |
| `make harvest NOTE=...` | Diff user edits and harvest line crops into spot buffer |
| `make label` | Launch local web labeling interface on port 8080 |
| `make train` | Run multi-epoch TrOCR fine-tuning across all batches |
| `make simmer` | Run fast 90s GPU burst on spot corrections with replay buffer |
| `make sweep` | Transcribe entire archive (`notepads/journals/*.pdf`) with manifest |
| `make clean-transcripts` | Run lexicon cleanup sweep across all transcripts in vault |
| `make charts` | Regenerate high-res CER, loss, and scaling law plots |
| `make clean` | Remove temporary caches and logs |

---

## Detailed Technical Documentation

* **[FINDINGS.md](FINDINGS.md)**: 17-section comprehensive technical report detailing PDF reverse engineering, line clustering geometry, benchmark curves, error taxonomies, and hardware profiling.
* **[BLOG_POST_OUTLINE.md](BLOG_POST_OUTLINE.md)**: Narrative essay structure for publication on [blog.lmorchard.com](https://blog.lmorchard.com).
* **[HANDOFF.md](HANDOFF.md)**: Engineering handoff and operational guide for agent and human developers.
