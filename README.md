# Personal Handwriting Recognition with TrOCR

An offline, personalized handwriting recognition pipeline to transcribe handwritten journals (Onyx BOOX e-ink tablet PDFs and physical paper notebook scans) into plain Markdown notes directly inside an Obsidian vault.

Fine-tuned on **2,721 personal handwriting lines** across 5 years of journals (2022–2026), achieving **2.27% Character Error Rate (CER)** on Apple Silicon M5 Max.

---

## The Daily Routine: Transcribe, Edit & Harvest

The daily workflow requires **zero arguments**—run two simple commands around your normal journaling:

```
[ BOOX Stylus Ink ] ──► [ notepads/journals/YYYY-MM-DD.pdf ]
                             │
                             ▼ (1. make transcribe: ~15s on Mac / ~60s on Linux)
                     [ notepads/transcriptions/YYYY/YYYY-MM-DD.md + .raw.md ]
                             │
                             ▼ (2. Review & Edit in Obsidian)
                     [ User-Edited Markdown ]
                             │
                             ▼ (3. make harvest: auto-diff, crop ink, git push)
                     [ handwriting-data/spot_corrections/ (on Forgejo) ]
                             │
                             ▼ (4. make simmer: 90s GPU burst when buffer fills)
                     [ Updated TrOCR Checkpoint ]
```

### 1. Transcribe Today's Entry

When you finish writing on your BOOX tablet and the PDF syncs to your vault:

```bash
make transcribe
```

* **Zero Arguments Needed**: Automatically scans `notepads/journals/*.pdf`, skips already transcribed files, and only processes new entries.
* **Vision Inference**: Runs line segmentation and TrOCR recognition (~15s on Mac GPU, ~60–90s on Linux CPU).
* **Vault Polish**: Automatically applies a Levenshtein clamp ($\le 2$) against your 166k-word personal Obsidian vocabulary to fix OCR typos without generative hallucination.
* **Obsidian Placement**: Writes the cleaned note to `notepads/transcriptions/YYYY/YYYY-MM-DD.md` with frontmatter linking back to the source PDF, and leaves a temporary `.raw.md` baseline alongside it.

---

### 2. Review and Edit in Obsidian

Open Obsidian and read through today's transcribed note.
* Fix any transcription typos directly in Markdown just like editing a normal note.
* Keep your authentic style, slang, and personal phrasing.

---

### 3. Harvest, Commit & Push

Once you're done editing in Obsidian:

```bash
make harvest
```

* **Zero Arguments Needed**: Automatically scans `notepads/transcriptions/` for notes with `.raw.md` baselines, detects your edits, crops the physical ink lines from the PDF, and appends them to `spot_corrections/dataset.csv`.
* **Auto Git Commit & Push**: Automatically commits the new training pairs and pushes them to your private Forgejo repository.
* **Cleans Up Baseline**: Removes the `.raw.md` file once corrections are safely banked.
* **Simmer Alert**: Reminds you when total spot corrections reach $\ge 50$ or $100$ lines.

---

## Working with Historical Entries

When working with older past volumes (rather than today's entry), bypass auto-detection by specifying the exact file:

### Case A: Re-transcribe an Old Entry with the Latest Model
To run our newest 2.27% CER model on an older journal:

```bash
# 1. Re-transcribe the specific PDF (recreates .raw.md and updates .md)
make transcribe PDF=2023-06-13.pdf

# 2. Edit in Obsidian to fix any remaining typos

# 3. Harvest against the freshly generated baseline
make harvest NOTE=notepads/transcriptions/2023/2023-06-13.md
```

### Case B: Fix Typos in an Already Transcribed Archive Note
If you're browsing Obsidian and fix typos in an existing older note:

```bash
# 1. Edit the note in Obsidian directly

# 2. Harvest directly (pulls raw baseline from historical backup tarball)
make harvest NOTE=notepads/transcriptions/2024/2024-05-12.md
```

---

## Simmer Retraining (Active Learning)

Whenever ~50–100 spot corrections accumulate in the simmer buffer:

```bash
make simmer
```

This launches a **90-second GPU simmer burst** on Apple Silicon M5 Max with an experience replay buffer, baking your recent corrections into the model without catastrophic forgetting.

---

## Setup & Configuration

1. **Clone Repositories**:
   Clone this code repo and your private data repo side-by-side:
   ```bash
   git clone git@github.com:lmorchard/handwriting-recognition.git
   git clone ssh://git@myriad-docker:22222/lmorchard/handwriting-data.git
   ```

2. **Configure Environment**:
   Copy `.env.example` to `.env` and set your local paths:
   ```bash
   HANDWRITING_DATA_DIR=../handwriting-data
   HANDWRITING_MODEL_DIR=../handwriting-data/models/best_model
   OBSIDIAN_VAULT_DIR=/Users/lmorchard/Documents/obsidian/main
   ```

3. **Install Dependencies**:
   ```bash
   make setup
   ```

---

## Command Reference

| Target | Description |
| :--- | :--- |
| `make transcribe` | Auto-discover and transcribe new journal PDFs into Obsidian |
| `make transcribe PDF=...` | Transcribe and clean a specific PDF volume |
| `make harvest` | Auto-detect edited notes, crop ink, and push to Forgejo |
| `make harvest NOTE=...` | Harvest corrections from a specific Markdown note |
| `make simmer` | Run fast 90s GPU burst on spot corrections with replay buffer |
| `make train` | Run full multi-epoch fine-tuning across all 30 batches |
| `make label` | Launch local web labeling interface on port 8080 |
| `make sweep` | Transcribe entire archive (`notepads/journals/*.pdf`) with manifest |
| `make clean-transcripts` | Run lexicon cleanup sweep across all transcripts in vault |
| `make charts` | Regenerate high-res CER, loss, and scaling law plots |
| `make clean` | Remove temporary caches and logs |

---

## Detailed Documentation & Technical Deep Dives

* **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Asymmetric homelab split (Linux VM CPU ingestion vs. Mac M5 Max GPU forge) and private data separation.
* **[docs/PDF_EXTRACTION.md](docs/PDF_EXTRACTION.md)**: BOOX vector `/Ink` decoupling from raster dot grids, geometric line clustering, and landscape two-column layout handling.
* **[FINDINGS.md](FINDINGS.md)**: Comprehensive 17-section technical report detailing empirical CER progression, error taxonomies, and hardware benchmarks.
* **[BLOG_POST_OUTLINE.md](BLOG_POST_OUTLINE.md)**: 6-Act narrative essay outline for [blog.lmorchard.com](https://blog.lmorchard.com).
* **[HANDOFF.md](HANDOFF.md)**: Operational runbook and engineering context for AI agents and human collaborators.
