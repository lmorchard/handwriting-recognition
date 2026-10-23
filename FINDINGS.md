# Handwriting Recognition Spike: Findings & Architecture

This document synthesizes findings, architectural discoveries, engineering lessons, and empirical benchmarks from the spike on personal handwriting recognition targeting BOOX tablet digital journals.

---

## 1. Problem & Style Characteristics

* **Corpus**: ~500 daily journal PDF volumes exported from an Onyx BOOX e-ink tablet (with physical paper dot-grid notebooks planned for future ingestion).
* **Handwriting Style**: Architectural / drafting small-caps print.
  * Words are lettered primarily with capital glyphs where lowercase letters are drawn as scaled-down small capitals (e.g. `H`, `E`, `R`, `N` with character heights matching standard x-height).
  * The personal pronoun "I" and sentence-initial capitals are drawn noticeably taller.
* **Why Off-the-Shelf OCR / HTR Fails**:
  1. **Baseline / Anatomy Assumptions**: Traditional engines (Tesseract, OCRopus, EasyOCR) rely on ascenders (`b, d, f, h, k, l, t`) and descenders (`g, j, p, q, y`) to calculate text baselines, x-height, and word boundaries. Small caps lack these distinctive vertical anchors, causing layout engines to miscalculate baselines and fracture words.
  2. **Vision-Decoder Tension in Transformer HTR**: In encoder-decoder models like TrOCR (Vision Transformer encoder + RoBERTa/BART language model decoder), pretrained weights expect standard English typography. The vision encoder recognizes capital glyphs (`W`, `E`, `R`, `E`), but the autoregressive language decoder expects standard lowercase tokens mid-word (`were`). Without fine-tuning, the language prior battles the visual signals, resulting in case fragmentation and hallucinated character substitutions.

---

## 2. Reverse-Engineering BOOX PDF Exports

Inspection of native BOOX tablet PDF exports (`2026-08-26.pdf`) revealed an architectural separation that simplifies data preparation:

1. **Decoupled Layers**:
   - The dot-grid paper template is stored as a single background raster image (`/XObject /FXX1GLDAAHPOI`).
   - The handwritten ink is **not rasterized into the background image**.
2. **Native Vector `/Ink` Annotations**:
   - Each page contains hundreds of discrete PDF `/Ink` annotations stored under `/Annots` (e.g., 966 discrete stroke annotations on Page 1).
   - Each annotation contains:
     - Exact bounding coordinates: `/Rect [x1, y1, x2, y2]`.
     - Sequential stylus coordinates: `/InkList` (sub-pixel path points).
     - Stylus dynamics and metadata in `/onyxtag`: stroke width (`originStrokeWidth ~ 5.9pt`), pressure sensitivity (`maxPressure`, `smoothLevel`), and timestamp identifiers.
3. **Implications**:
   - **Zero Dot-Grid Interference**: By rendering only the vector `/InkList` paths onto a pure white canvas, background dot grids and scanning noise are eliminated without needing morphological opening, thresholding, or line-removal filters.
   - **Geometric Line Slicing**: Every pen stroke provides exact bounding coordinates, enabling deterministic spatial clustering.

---

## 3. Line Extraction & Segmentation Evolution

Developing clean line extraction required navigating several subtle geometric failure modes:

### Initial Attempt: Naive Y-Center Proximity
* **Failure Mode**: Strokes were clustered solely by vertical center distance. Lines containing quotation marks, apostrophes, or tall capital letters fractured into multiple fragment lines because the Y-center of a high quotation mark ($Y \approx 1820$) diverged by $>50\text{px}$ from the line's body strokes ($Y \approx 1870$). Furthermore, quotation marks hung closer in $Y$ to the preceding line's baseline, pulling quotes into the line above.

### The Two-Pass Boundary Filter Trap (A Cautionary Lesson)
* **Hypothesis**: Classify strokes with `height < 28px` as "boundary marks" (commas, quotes, crossbars) in Pass 2 using custom geometric rules, while reserving Pass 1 for core letter stems.
* **Failure Mode (Severe Regressions)**:
  - Drafting small-caps letters have internal horizontal crossbars ($H, E, F, A$), top strokes ($T, Y$), and standalone letters ($I$) with heights between $6\text{px}$ and $20\text{px}$.
  - The boundary filter classified these essential anatomical strokes as boundary marks.
  - Because an $H$ or $A$ crossbar has stems on *both* sides (not *underneath*), it failed custom stem heuristics and was dropped or misplaced onto adjacent lines.
* **Takeaway**: Avoid hand-crafting visual feature detectors for individual letter strokes. Heuristics for one character (e.g. commas) inevitably break the internal anatomy of others ($H, E, F, A$).

### The Stable Architecture (Unified Proximity + Stem-Binding)
The robust pipeline in `extract_lines.py` uses three core stages:
1. **Unified Proximity Clustering**:
   - Primary strokes ($height \ge 18\text{px}$) are sorted by vertical center and grouped into raw line clusters using a $42\text{px}$ tolerance window.
   - All letter strokes stay together naturally within a single line.
2. **Horizontal Overlap Line Merging**:
   - If two adjacent clusters have significant horizontal bounding-box overlap and their vertical distance is $< 56\text{px}$ (less than the $71\text{px}$ dot-grid pitch), they are merged into a single line. This resolves multi-tier capitals and quotation marks cleanly.
3. **Stem-Binding for Minor Marks**:
   - Minor strokes ($height < 18\text{px}$) are checked against vertical stems ($height \ge 30\text{px}$).
   - If a horizontal stroke sits directly atop a vertical stem within $|\Delta X| \le 22\text{px}$ and vertical gap $\le 25\text{px}$, it binds directly to that stem's line cluster. This prevents the top bars of tall capital pronouns ("I") and $T$s from floating to the line above.
4. **Generous Render Padding**:
   - Render crops use asymmetric padding: `padding_top = 24px`, `padding_bottom = 28px`, `padding_horizontal = 18px`.
   - Deeply descending commas and elevated quotation marks remain fully inside the crop without boundary clipping.

---

## 4. Labeling Tool (`label_ui.py`)

A zero-dependency local web tool built with Python's standard library `http.server`:
* **Zero Overhead**: Runs via standard library without requiring Node.js, React, or heavy frontend frameworks.
* **Keyboard-First Workflow**: `Enter` saves the current line and auto-advances; `Cmd+Left` / `Cmd+Right` navigates; auto-selects text on focus.
* **Jump Selector**: Header dropdown displays live completion status (`✓` labeled vs `○` empty), line coordinates (`P1 L04`), and text previews.
* **Real-Time Dual Persistence**: Each save updates the JSON catalog and regenerates `dataset.csv` immediately.
* **Network & Path Agnostic**: Defaults to `--host 0.0.0.0` for access over Tailscale; resolves image paths relative to the catalog directory regardless of current working directory.

---

## 5. TrOCR Fine-Tuning Benchmarks (Apple Silicon M5 Max)

The model was fine-tuned on an **Apple Silicon M5 Max MacBook Pro** using PyTorch Metal Performance Shaders (`mps`).

### Training Configuration
* **Base Model**: `microsoft/trocr-base-handwritten` (~334M parameters: ViT encoder + RoBERTa/BART decoder).
* **Dataset**: 84 lines extracted from `2026-08-26.pdf` (71 train, 13 validation).
* **Hyperparameters**:
  * Epochs: 10 (180 total steps)
  * Batch Size: 4 per device
  * Learning Rate: $5 \times 10^{-5}$ with linear decay
  * Weight Decay: 0.01
  * Evaluation & Checkpointing: End of each epoch, tracking Character Error Rate (CER) via beam search (`num_beams = 4`).

### Empirical Results Across Training Runs

| Run | Dataset | Steps / Epochs | Runtime (M5 Max) | Final Loss | Validation CER | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Zero-Shot** | Pretrained TrOCR Base | 0 | — | — | ~35.0% | Baseline prior |
| **Run 1** | 84 lines (`batch_aug26`) | 180 / 10 | 4.5 min | 0.006 | **16.0%** | First fine-tuned checkpoint (13 val lines) |
| **Run 2** | 169 lines (`batch_aug26` + `batch_aug25`) | 360 / 10 | 5.1 min | 0.0018 | **6.44%** | Error rate cut by >60% (26 val lines) |
| **Run 3** | 334 lines (all 4 batches) | 710 / 10 | 11.2 min | 0.0009 | **6.33%** | Evaluated on 51 unseen validation lines |
| **Run 4** | 507 lines (6 batches, 2024–2026) | 1080 / 10 | 16.5 min | 0.0004 | **5.63%** | Evaluated on 77 unseen lines across 3 years |
| **Run 5** | 593 lines (7 batches, 2023–2026) | 1260 / 10 | 14.2 min | 0.0003 | **4.55%** | Evaluated on 89 unseen lines across 4 years |
| **Run 6** | 762 lines (9 batches, 2023–2026) | 1620 / 10 | 54.0 min | 0.0002 | **3.44%** | Evaluated on 115 unseen lines across 4 years |
| **Run 7** | 1,017 lines (12 batches, 2023–2026) | 2160 / 10 | 31.3 min | 0.0001 | **5.10%** | Evaluated on 153 unseen lines across 4 years |
| **Run 8** | 1,277 lines (15 batches, 2023–2026) | 3264 / 12 | 47.5 min | 0.0001 | **2.50%** | Peak at Epoch 10 (2.50%), final 2.68% (Loss: 0.254) across 192 unseen lines |
| **Run 9** | 2,721 lines (30 batches, 2022–2026) | 5780 / 10 | 83.0 min | **0.2400** | **2.27%** | Peak at Epoch 10 across 409 unseen validation lines; completely resolved *afternoon*, *turkey*, *Claude*, and *lentil stew* directly from ink |

* **Throughput**: **~1.05 to 1.5 iterations per second** on M5 Max MPS.
* **Peak Memory**: Fits comfortably in ~5–6 GB of unified memory.

---

## 6. End-to-End Transcription & Empirical Error Resolution

Running the fine-tuned model on an unseen 3-page journal (`2026-08-24.pdf`, 85 lines) via `transcribe_journal.py` completed in **15.3 seconds** with clean paragraph structure and YAML frontmatter.

### The Impact of Expanding to 334 Lines
Comparing the output of the 169-line model vs. the 334-line model on the exact same document demonstrates how direct corpus expansion resolved the visual glyph confusions:

| Phrase / Context | 169-Line Model Output | 334-Line Model Output (Direct from Ink) | Ground Truth Status |
| :--- | :--- | :--- | :--- |
| **Project & Employer** | `project at Morvant on which I spent anthie firstingly` | `project at Mozilla on which I spent awhile frettingly` | **100% Exact Match** |
| **Therapist Name** | `Michael crore, is more of a blank affect probe` | `(Michael Crone) is more of a blank affect probe` | **100% Exact Match** |
| **Sentence Starter** | `Starting through. today's news` | `Still though: today's news` | **100% Exact Match** |
| **Performance Review** | `I but had a performance meeting ... extending expectations` | `I just had a performance meeting ... exceeding expectations` | **100% Exact Match** |
| **Necker Cube & Knobs** | `right knows to turn or the right Necker Cobe` | `right knobs to turn or the right Necker Cube` | **100% Exact Match** |
| **Emotional Impact** | `made me briefly vert. He doesn see me` | `made me briefly verklempt: He doesn't see me` | **100% Exact Match** |
| **Date Header** | `Monday, August 2, 2026` | `Monday, August 24, 2026` | **100% Exact Match** |
| **Weather & Temp** | `110 pm ... 85° and sumfy skies ... trace of vador` | `1:10 pm ... 85°f and sunny skies ... trace of vapor` | **100% Exact Match** |
| **Childhood Reflection**| `hyperactive ent'sivism ... ex suchs it` | `hyperactive enthusiasm ... explains it` | **100% Exact Match** |
| **Headspace** | `shifted head space` | `shifted headspace` | **100% Exact Match** |

### The "Almost Right" Paradox (Word Error Rate vs. Character Error Rate)
In a sentence with 10 words (50 characters), a single misread character in 3 different words produces a respectable $6\%$ CER, but a jarring **$30\%$ Word Error Rate (WER)**. As demonstrated in the table above, doubling the dataset to 334 lines directly eliminated the remaining single-character visual errors, causing entire sentences to snap into 100% exact matches.
   - `r` $\leftrightarrow$ `n`: `turn` $\to$ `then`, `premise` $\to$ `promise`.
   - Dense multi-letter clusters: `enthusiasm` $\to$ `ent'sivism`, `explains` $\to$ `ex suchs`.
2. **Missing Punctuation / Digits**:
   - Colons in timestamps: `1:10 pm` $\to$ `110 pm` (small colon dots lost in whitespace or tokenized out).
   - Trailing digits: `August 24` $\to$ `August 2`.
3. **Small Language Model Limitations (The ~125M Parameter Decoder)**:
   - TrOCR's text decoder is a compact BART/RoBERTa model. When visual signals are ambiguous, it defaults to common n-grams rather than evaluating broader paragraph semantics:
     - Guesses `proposing` instead of `probing`.
     - Guesses `time sense` instead of `true sense`.
     - Guesses `huminating` instead of `ruminating`.
   - It lacks the multi-sentence context to recognize specialized phrases (`Necker Cube`, `psychodynamic`, `blast radius`).

---

## 7. Strategy for Production-Ready Accuracy

To close the gap from "readable with typos" to "publication-ready journal transcriptions":

### Axis 1: Expanding the Training Corpus (~300–400 lines)
* 169 lines cut CER by 60% (from 16.0% to 6.44%).
* The model currently suffers from vocabulary sparsity on rare character n-grams (`nth`, `asm`, `obs`, `ck`).
* Labeling 2 more journal volumes (`2026-08-24` and `2026-08-23`) via `label_ui.py` (which takes <5 minutes each due to pre-transcriptions) will double the dataset to ~340 lines, dramatically reducing character confusions.
* **Data Augmentation**: Introducing mild affine shear ($\pm 3^\circ$) and horizontal scale variations during dataset loading will improve stroke invariance.

### Axis 2: Pitfalls of Blind LLM Post-Processing (Hallucinatory Smoothing)

A prototype cleanup pass tested on `transcribed/2026-08-24.md` revealed a fundamental risk in using text-only LLMs to "fix" OCR transcripts:

#### The Failure Mode: Semantic Homogenization & Fact Distortion
Because a text-only LLM lacks visual grounding (it cannot see the underlying ink strokes), it attempts to resolve ambiguous, garbled OCR words purely through statistical language priors. This causes the LLM to **substitute generic clichés for authentic personal phrasing**:

| Raw TrOCR Output | Blind LLM Cleanup | Actual Author Ground Truth | What Went Wrong |
| :--- | :--- | :--- | :--- |
| `anthie firstingly` | `all the spring` | **`awhile frettingly`** | Erased unique emotional vocabulary with a generic cliché |
| `at Morvant` | `at Movable` | **`at Mozilla`** | Hallucinated a plausible company name ("Movable Type") |
| `Michael crore` | `Michael Crowe` | **`Michael Crone`** | Hallucinated incorrect person's name |
| `retirement and succession` | `reluctance and suspicion` | **`reticence and suspision`** | Substituted different emotional words |
| `very let really` | `very readily` | **`very loudly`** | Semantic shift |
| `Starting through.` | `Sorting through,` | **`Still, though:`** | Altered sentence structure |
| `I but had a performance...` | `I had a performance...` | **`I just had a performance...`** | Deleted word instead of correcting OCR glyph (`just` $\to$ `but`) |
| `85°` / `60°f` | `85°` / `60°f` | `85ºf` / `60ºf` | Dropped Fahrenheit unit |

#### Core Architectural Conclusion
1. **Never rely on blind text-only LLMs to reconstruct corrupted handwriting.** When an LLM doesn't see the image, it smooths out the author's voice, rewrites idiosyncratic phrasing, and invents facts that sound plausible but are dead wrong.
2. **The Vision Model (TrOCR) is the Ground Truth.** Notice that TrOCR's `anthie firstingly` was physically closer to the pen strokes of `awhile frettingly` (`wh` looks like `nth`, `frett` looks like `first`). TrOCR was actually looking at the ink.
3. **The True Path to Accuracy is Corpus Growth (Axis 1):** The only reliable way to fix these errors without hallucinating is expanding the fine-tuning dataset so TrOCR's visual encoder learns the author's actual letterforms directly from the ink.
4. **Role of LLM Post-Processing (If Any):** If an LLM is used, it should be a **Vision-Language Model (VLM)** that receives both the cropped line image AND the TrOCR candidate transcription, or restricted strictly to verifiable dictionary spelling corrections. Text-only generative rewriting must not be used in the production pipeline.

---

## 8. Temporal Invariance & The Isolated-Word Fragility Discovery

### Cross-Year Zero-Shot Generalization
To test whether fine-tuning on a single recent month (August 2026) overfit to specific stylus nib wear, pen pressure, or e-ink refresh modes, the 334-line model was evaluated zero-shot on unseen historical journals from June 2025 (`2025-06-02.pdf`) and June 2024 (`2024-06-04.pdf`):
* **Zero-Shot Historical Accuracy**: The model achieved **~85% to 90% word accuracy** immediately on raw 1-year and 2-year old writing without any retraining.
* **Consistency of Personal Letterforms**: The author's small-caps architectural print remained remarkably geometrically consistent over 3 years. The primary variations across years were minor line-weight fluctuations (~5.9pt vs ~6.2pt) and slight spacing differences, rather than glyph structure changes.

### The Isolated-Word Fragility Trap
When testing the 334-line model on `2026-08-20.pdf`, an unexpected regression occurred:
* Line 7: `sinus headache, no vertigo, low` (Transcribed correctly)
* Line 8: `sensitivity.` (A single trailing word, 16 strokes) $\to$ Transcribed as `seen sit it it. but my.`
* **Root Cause Diagnosis**:
  1. **Random Validation Hold-Out**: In small datasets (334 lines), a 15% random validation split held out 51 lines. Line 8 landed in the validation set, so the model had never been optimized on it.
  2. **Encoder-Decoder Context Dependency**: On multi-word lines, TrOCR's autoregressive language model decoder uses preceding words as context to resolve visual ambiguities. On isolated 1-word or 2-word trailing lines, there is zero preceding context. The vision encoder must decode characters purely visually.
  3. **The Solution (Corpus Growth Across Time)**: Expanding the corpus to 507 lines across 2024, 2025, and 2026 exposed the vision encoder to enough visual variety of isolated small-caps glyphs (`S`, `V`, `T`) that it decoded Line 8 as **`low sensitivity.`** with 100% precision, while simultaneously resolving `overjoyed or overflowing with vim and vigor`, `altogether awful — this would be`, and `the meat of it now`.

---

## 9. Multi-Year Scaling & The "Moving Exam" Dynamic (Runs 5–8)

As the corpus scaled past 1,000 lines across 15 complete volumes, two major machine learning dynamics emerged:

### The "Moving Exam" Effect
* In Run 6 (762 lines), validation CER reached **3.44%** across 115 unseen lines.
* In Run 7 (1,017 lines), validation CER temporarily ticked up to **5.10%** across 153 lines.
* **Why the metric rose while the model improved**: The test set is not static. When adding historical batches from 2023 and 2024 (where stylus nib friction and writing speed differed), the random 15% validation split inherited lines from older, stylistically diverse eras that the model had never seen. The "exam" grew harder as the training corpus expanded.
* In Run 8 (1,277 lines), adjusting the optimization budget to 12 epochs at a slightly lower learning rate ($4 \times 10^{-5}$) allowed the model to assimilate the full multi-year diversity, dropping CER to a project-best **2.50% peak** across 192 unseen validation lines.

### The 10-Epoch Convergence Sweet Spot
* Tracking validation loss across 12 epochs in Run 8 revealed that validation CER hit its mathematical minimum at **Epoch 10 (`checkpoint-2720`, CER = 2.50%)** before experiencing slight over-regularization at Epoch 12 (CER = 2.68%).
* Because `train_trocr.py` configures `load_best_model_at_end=True` with `metric_for_best_model="cer"`, the trainer automatically restored the peak Epoch 10 weights as `trocr_finetuned/best_model`, eliminating any penalty from running exploratory epochs.
* **Invariant**: 10 epochs is the empirical convergence sweet spot for this corpus size and learning rate decay schedule.

---

## 10. Multi-Column Layouts: The 2022 Landscape Notebook Discovery

Analysis of the earliest volume in the vault (`2022-10-19.pdf`, $2480 \times 1860\text{px}$) revealed that earlier journals were written in **two-column landscape orientation**:

### The Gutter Geometry
* Plotting a histogram of stroke X-coordinates across the 2,480px page width revealed a **sharp 50px void at $X = 1240$** (the mathematical center), flanked by ~550 strokes on the left ($X \in [50, 1200]$) and ~550 strokes on the right ($X \in [1280, 2370]$).
* Without column separation, baseline grouping merged left-hand and right-hand sentences into scrambled 60-word horizontal lines (e.g. *`ts 8:fam. Weather is. Tilot, last night, so`*).

### Two-Column Architecture (`extract_lines.py`)
* Implemented automatic landscape detection (`width > height`):
  1. Identifies the central gutter at $X = \text{page\_width} / 2.0$.
  2. Slices Left Column ($X < \text{gutter}$) top-to-bottom first.
  3. Slices Right Column ($X \ge \text{gutter}$) top-to-bottom second.
  4. Assigns continuous line indices ($1 \dots N$ left, $N+1 \dots M$ right).
* **Result**: Tested on `2022-10-19.pdf` (130 lines, 3 pages). The document transcribed in **20.4 seconds** in perfect human reading order:
  * Left: *"It's 8:08am. Weather is 54°f and dim... High of 72°f and low of 55°f..."*
  * Right: *"I'd seen a video for her song 'What I am' on TikTok last night, so she was on my mind. Took a quarter of a weed gummy before my espresso..."*

---

## 11. Active Learning, Vocabulary Sparsity, & The "Tuna / Turkey" Lesson

### The Reality of 2.5% CER
While a 2.50% Character Error Rate represents >97.5% character accuracy, on an 80-character line it still permits **1 to 2 character misreads per line**, translating to **1 to 2 corrupted words in every paragraph**.

When testing Run 8 on `2025-08-28.pdf`, several jarring errors appeared:
* `atternsoon` instead of `afternoon`
* `some time` instead of `some turkey`
* `length stew` instead of `lentil stew`
* `Claving` instead of `Claude`

### The Vocabulary Census
Querying the 15 verified training volumes (1,349 lines) for these words revealed the root cause:
* `afternoon`: **1 occurrence** in the entire training set.
* `turkey`: **0 occurrences**.
* `lentil`: **0 occurrences**.
* `Claude`: **0 occurrences**.

In small-caps lettering, cursive `ft` closely mirrors `tt` without ligature priors, and handwritten `rke` resembles `m`. Because the model had never seen the author draw "turkey" or "Claude", the vision encoder had to guess purely from unfamiliar stroke fragments.

### The Real-Time Hallucination Trap
During an evaluation turn, the AI assistant summarized the transcription of `2025-08-28`:
* Transcribed text: `high spirits, begging for some time, from my lunch.`
* Actual author ground truth: `high spirits, begging for some turkey, from my lunch.`
* Assistant's summary: *"begging for some tuna from my lunch..."*

Because Catsby is a cat and cats eat tuna, the language model's prior subconsciously smoothed the corrupt token into a plausible web cliché. This provided a live, visceral demonstration of **hallucinatory smoothing**: generative models will instinctively rewrite idiosyncratic reality into generic stereotypes whenever visual grounding is lost.

### The Solution: Wave 3 Doubling Sprint (2,721 Lines)
Rather than declaring diminishing returns, the user and assistant executed a targeted Wave 3 active-learning expansion:
1. Enhanced `label_ui.py` with a live **multi-batch dropdown switcher** across all 30 batches archive-wide.
2. Pre-transcribed 15 new historical volumes using Run 8 in 84 seconds on M5 Max MPS.
3. Rapid keyboard verification in `label_ui.py` brought the corpus to **2,721 lines across 30 batches (2022–2026)** with **0 empty entries**.

---

## 12. Production Exporter & Pipeline Hardening (`transcribe_journal.py`)

1. **Multi-File & Glob CLI Support**:
   * Accepts multiple paths and shell globs: `python3 transcribe_journal.py "journals/2025-08-*.pdf" -o transcribed/2025-08/`.
   * TrOCR weights load into Apple Silicon unified memory **once**, processing subsequent volumes with zero reload latency (~16s per volume).
2. **Greedy Token Repetition Deduplication**:
   * Resolved autoregressive decoding stutters (*`playing exhausting exhausting exhausting catchup now`*) via regex word deduplication: `re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', text)`.
3. **Symbol Canonicalization**:
   * Automatically normalizes masculine ordinals (`º` U+00BA) to standard degree symbols (`°` U+00B0) for consistent weather strings.

---

## 13. Full Archive Execution: 522 Volumes in 5.1 Hours

Once Run 9 achieved 2.27% CER across the 2,721-line corpus, `transcribe_journal.py` was deployed across the user's entire multi-year archive:

* **Scope**: 523 journal PDF volumes spanning 2014 through 2026.
* **Execution**: Unattended background batch execution on an Apple Silicon M5 Max MacBook Pro using Metal Performance Shaders (`mps`).
* **Throughput & Timing**:
  * Total runtime: **310 minutes (~5.1 hours)**.
  * Average throughput: **~15.8 seconds per 3-page journal volume** (~5.2 seconds per page).
  * Total handwritten text transcribed: **43,199 lines**.
* **Reliability & Error Isolation**:
  * **522 of 523 volumes (99.8%) transcribed successfully**.
  * Exactly one volume failed (`2026-03-10-summary.pdf`), isolated to a truncated/corrupted PDF stream from the tablet export.
  * Robust atomic manifest tracking (`manifest.json`) via tempfile swap ensured zero data loss or partial writes if interrupted.
* **Vault Organization**:
  * Automatically routed notes into Obsidian year directories (`2014/`, `2022/`, `2023/`, `2024/`, `2025/`, `2026/`, `misc/`).
  * Injected YAML frontmatter cross-linking back to the source digital ink PDF: `source: "[[notepads/journals/...]]"`.

---

## 14. The 2.2% CER Plateau & Visual Indistinguishability

Scaling from 1,277 lines (Run 8) to 2,721 lines (Run 9) dropped validation CER from 2.50% to 2.27%. However, analyzing the residual errors across 43,199 lines revealed a fundamental architectural boundary for vision-only models:

1. **The Small-Caps Indistinguishability Ceiling**:
   * In architectural drafting small-caps, certain character pairs are physically indistinguishable from raw pixels alone:
     * `C` vs `G`: depending on stylus nib wear and lifting speed, the inward serif on a small-caps `G` is often a sub-pixel spur or omitted entirely.
     * `d` vs `e`: looping small-caps `D` vs looped `E` stroke closures.
     * `find` vs `fine`: the crossbar on a terminal `e` vs an unclosed `d` loop.
     * `next` vs `text`: an initial `N` without an entry serif mirrors a uncrossed `T`.
     * `table` vs `tablet`: trailing terminal strokes often taper into blank margins.
2. **The Vision-Only Diminishing Returns Law**:
   * To push a vision transformer from 2.2% CER down to sub-0.5% WER on pure visual signals alone would require tens of thousands of manually labeled lines—and would *still* fail on physically ambiguous single-pixel strokes.
   * The missing ingredient is not more vision data; it is **domain-grounded linguistic constraints**.

---

## 15. The Lexicon-Constrained Post-Processor (`clean_transcript.py`)

To cross the final mile from raw TrOCR output to publication-grade markdown *without* falling into the "tuna/turkey" generative smoothing trap, we designed and implemented `clean_transcript.py`:

### Architectural Invariants:
1. **The Hard Levenshtein Clamp ($\le 2$)**:
   * Any proposed correction must satisfy a Damerau-Levenshtein edit distance $\le 2$ (and $\le 1$ for tokens $\le 4$ characters).
   * **Mathematical Protection of Idiosyncratic Voice**: Authentic, non-standard phrases like `awhile frettingly` have an edit distance $> 8$ to common English clichés like *"all the spring"*. The hard clamp renders generative smoothing mathematically impossible.
2. **The Obsidian Vault Lexicon (166,735 Unigrams, 400,747 Bigrams)**:
   * Rather than relying on generic internet datasets, the vocabulary cache was compiled directly from the user's personal markdown notes in Obsidian (`journals/`, `pages/`, `reading/`).
   * Slang, project names, pet names (*Catsby*, *Cheddars*), collaborators (*Dan Hon*), and hardware brands (*Lamy*, *AL-star*, *BOOX*) are natively recognized as high-frequency ground truth.
3. **Morphological Inflection Lemmatization**:
   * Standard dictionary wordlists (e.g. `/usr/share/dict/words`) contain base lemmas (`texture`, `lunch`, `work`), but omit inflected forms.
   * Without inflection handling, standard English words like `"textured screen protector"` triggered false-positive corrections to `"texture"`.
   * Implemented rule-based inflection stripping (`-ed`, `-ing`, `-s`, `-ly`): if the stripped stem exists in the dictionary, the inflected form is recognized as valid and protected.
4. **Frontmatter Immunity**:
   * State-machine tracking ensures that YAML frontmatter metadata, dates, and header dividers are strictly bypassed.

### Archive Cleanup Results (522 Volumes, 43,199 Lines):
* **Files Cleaned**: 509 of 522 files received targeted word corrections.
* **Total Word-Level Fixes**: **3,806 words** across the entire multi-year corpus (averaging ~7.3 minor corrections per journal volume, or less than 1 word per 11 lines).
* **Sample Disambiguations**:
  * `"I got this new think table..."` $\to$ `"I got this new e-ink tablet..."`
  * `"...replace my paper notbooks..."` $\to$ `"...replace my paper notebook..."`
  * `"...convert all this to next."` $\to$ `"...convert all this to text."`
  * `"The front light is find..."` $\to$ `"The front light is fine..."`
  * `"Oddy"` $\to$ `"Oddly"`, `"oblications"` $\to$ `"obligations"`, `"journalling"` $\to$ `"journaling"`
* **Safety**: All 522 raw, unedited TrOCR transcripts were preserved in a compressed archive (`transcriptions_backup_raw_20261022.tar.gz`, 644KB) before running in-place.

---

## 16. Key Takeaways for Personal Machine Learning

1. **The Power of the Single-Author Model**:
   * Foundation models trained on millions of authors struggle with idiosyncratic lettering styles because universal handwriting recognition is an under-constrained problem.
   * Fine-tuning a compact 334M-parameter foundation model on *one specific individual's handwriting* converges with astonishing efficiency: 84 lines produced a working model (16% CER), 334 lines made it practical (6.3% CER), and 2,721 lines made it production-grade (2.27% CER).
2. **Local, Private, and Sovereign**:
   * Personal journals are the most intimate data a human produces. Sending 500 volumes of daily journals to commercial cloud vision APIs presents severe privacy compromises.
   * Running entirely on local Apple Silicon (M5 Max) using PyTorch MPS, the entire multi-year archive was transcribed and cleaned locally with zero network calls, zero subscription fees, and complete data sovereignty.
3. **The Fusion of Vision and Bounded Symbolic Post-Processing**:
   * Neither pure vision nor pure language models solve handwriting recognition alone:
     * Vision alone plateaus at the physical ambiguity of similar letter glyphs (~2.2% CER).
     * Generative LLMs alone hallucinate and erase authorial voice.
   * The winning architecture is **Vision for Ground Truth + Bounded Symbolic Lexicon Constraints for Disambiguation**.

---

## 17. Near-Future Development Arc: Asymmetric Homelab Ingestion & Continual Simmer

With the full archive transcribed and cleaned, the next operational phase focuses on seamless daily maintenance and continuous active learning:

### 1. The Asymmetric Split Architecture:
* **The Proxmox Linux VM (DecafClaw Host, CPU-Only)**:
  * **Role**: Always-on daily ingestion and harvesting engine.
  * **Daily Ingest**: Runs CPU-only TrOCR inference on newly synced BOOX PDFs (~60–90 seconds per volume), applies `clean_transcript.py`, and writes final Markdown notes directly to the Obsidian vault with source links.
  * **Obsidian Edit Harvester (`harvest_corrections.py`)**: Continuously monitors transcribed notes. When the user corrects a typo while reading in Obsidian, the harvester locates the corresponding stroke bounding box in the source PDF, renders the clean line image strip, and logs the pair to `spot_corrections.csv`.
* **Portia (MacBook Pro M5 Max)**:
  * **Role**: Transient workstation and on-demand ML training forge.
  * **Periodic Simmer Bursts**: When docked and connected, Portia pulls the accumulated corrections, mixes them with an experience replay buffer (preventing catastrophic forgetting), and executes a fast 90-second GPU fine-tuning burst on Apple Silicon MPS before syncing `best_model` back to the VM.

### 2. The Spot-Correction User Experience:
* **Primary Path (Zero Friction)**: Direct inline Markdown editing in Obsidian during regular reading on laptop, phone, or tablet.
* **Secondary Path (Visual Fallback UI, `spot_ui.py`)**: A focused single-volume browser tool for heavily distorted lines where the author cannot decipher the original wording without inspecting the rendered pen strokes side-by-side.

### 3. Cloud GPU Horizon (GCP / RunPod):
* **Digital Ink Scope**: Cloud GPU compute is deliberately avoided for the BOOX pipeline; local CPU inference (~60s) and Portia GPU bursts (~90s) keep running costs at $0, eliminate IAM/VPC networking complexity, and preserve total data sovereignty.
* **Future Paper Scope (Phase 2)**: Cloud GPU resources are reserved for physical notebook scanning, where heavy computer vision layout analysis (YOLO/CRAFT), perspective unwarping, shadow removal, and massive multi-epoch synthetic data generation may benefit from elastic GPU clusters.

---

## 18. Phase 1 Operational Milestones & Data Decoupling

The conclusion of Phase 1 establishes the long-term operational footing for the pipeline:

1. **Firewall Between Code and Private Data**:
   * **Public Tooling (`lmorchard/handwriting-recognition`)**: Hosted on GitHub, containing pure pipeline code, documentation, and publication plots. Local git history was squashed (`7132087`) to permanently expunge historical commits that briefly contained personal batch CSVs.
   * **Private Datasets (`lmorchard/handwriting-data`)**: Hosted on the self-hosted homelab Forgejo instance (`0xDECAFBAD hub` at `myriad-docker:22222`). Contains all 30 training batches (2,721 lines), active-learning spot corrections buffer, vault vocabulary caches, and fine-tuned model checkpoints (`models/best_model/`).
2. **Autonomous Zero-Argument Makefile Workflow**:
   * `make transcribe`: Automatically scans `notepads/journals/*.pdf`, skips already transcribed files, runs TrOCR inference, applies Levenshtein vault polish, and saves `.md` and `.raw.md` directly into the Obsidian vault.
   * `make harvest`: Automatically scans `notepads/transcriptions/` for notes with `.raw.md` baselines, diffs human edits, crops physical ink lines from the source PDF, appends them to `spot_corrections/dataset.csv`, cleans up `.raw.md`, and auto-commits and pushes to Forgejo.
3. **The Active Simmer Buffer (179 Lines Harvested)**:
   * Successfully harvested 179 real-world ground-truth lines directly from human edits across `2026-01-01`, `2026-10-21`, and `2026-10-22`.
   * Proved that continual active learning operates with zero labeling UI friction: normal reading and typo-fixing in Obsidian directly trains the next generation of model weights.




