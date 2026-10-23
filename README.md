# Getting a computer to turn my handwriting into text

Personal handwriting recognition project for handwritten journals (BOOX e-ink tablet PDFs and physical paper notebook scans).

My handwriting uses an architectural/drafting small-caps style that trips up off-the-shelf OCR engines. This project builds a personalized pipeline centered on fine-tuning Microsoft TrOCR on line-level crops.

## Key Discovery: BOOX PDFs

BOOX tablets export the page template (dot grid) as a raster background image (`/XObject`), but export all handwritten strokes as native vector PDF `/Ink` annotations (`/Annots`).

This means:
1. We can extract pure vector ink paths on a clean white background with zero dot-grid interference.
2. Strokes have exact bounding coordinates, so line segmentation is computed geometrically by clustering stroke vertical centers—no Tesseract, EasyOCR, or heuristic contour slicing required.

---

## Pipeline Workflow

### Step 1: Extract Line Images from BOOX PDFs

Run the vector extraction script on one or more journal PDFs:

```bash
uv run --with pypdf,pillow,numpy python3 extract_lines.py \
  /path/to/journal.pdf \
  --output-dir dataset_run1 \
  --pages 1 2 3
```

This generates:
- `dataset_run1/images/`: Individual clean line strip PNGs (`..._p01_line01.png`, etc.).
- `dataset_run1/<pdf_name>_lines.json`: Line metadata catalog.

### Step 2: Annotate Ground Truth (Local Web UI)

Launch the zero-dependency local labeling tool:

```bash
# Listen on all interfaces (default 0.0.0.0) so you can access over Tailscale:
python3 label_ui.py --catalog dataset_run1/<pdf_name>_lines.json --port 8080

# Or explicitly bind:
python3 label_ui.py --catalog dataset_run1/<pdf_name>_lines.json --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080` (or `http://decafclaw:8080` over Tailscale) in your browser:
- Header dropdown jump selector with live progress checkmarks (`✓` / `○`) and line text previews.
- Displays line image crop with keyboard autofocus.
- Type transcription in standard sentence casing (`Today I went...`).
- Press <kbd>Enter</kbd> to save and advance; <kbd>Cmd+Left</kbd> / <kbd>Cmd+Right</kbd> to navigate.
- Automatically saves progress to `<pdf_name>_lines.json` and syncs to `dataset_run1/dataset.csv`.

### Step 3: Fine-Tune TrOCR on Apple Silicon / GPU

Fine-tune `microsoft/trocr-base-handwritten` across one or more batches using Hugging Face `Seq2SeqTrainer` and Apple Silicon Metal Performance Shaders (`mps`):

```bash
uv run --with-requirements requirements.txt python3 train_trocr.py \
  --batches batch_aug26 batch_aug25 \
  --epochs 10 \
  --batch-size 4
```

- Automatically detects and uses Apple Silicon GPU (`mps`) on macOS (~1.5 it/s, 5.1 minutes for 10 epochs across 169 lines on M5 Max).
- Evaluates Character Error Rate (CER) per epoch (achieved **6.44% CER**) and saves the best model to `trocr_finetuned/best_model`.

### Step 4: Transcribe Entire Journal Volumes to Markdown

Run end-to-end automated transcription on any raw BOOX PDF:

```bash
# Transcribe directly to Markdown:
uv run --with-requirements requirements.txt python3 transcribe_journal.py path/to/journal.pdf

# Or specify custom output path:
uv run --with-requirements requirements.txt python3 transcribe_journal.py path/to/journal.pdf -o vault/journals/2026/2026-08-24.md
```

- Extracts vector lines in-memory without saving scratch files.
- Detects paragraph spacing and thematic divider rules (`---`).
- Normalizes punctuation and contraction spacing.
- Transcribes a full 3-page volume (~85 lines) in **~15 seconds** on Apple M5 Max.

---

## Detailed Findings & Benchmarks

See [FINDINGS.md](FINDINGS.md) for in-depth documentation on:
- BOOX PDF internal vector structure.
- Line segmentation geometry (clustering, line merging, stem-binding).
- Avoiding rule-based boundary filter regressions.
- Quantitative benchmarks and qualitative validation outputs on the M5 Max.
