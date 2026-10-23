# Blog Post Outline: Teaching a Local Vision Model to Read My Own Handwriting

> **Target Venue**: [blog.lmorchard.com](https://blog.lmorchard.com)  
> **Voice & Tone**: Curious tinkerer thinking in public; candid, technical without performing expertise; personal stakes, homelab realities, cats, and the stubborn pleasure of making computers do things locally.  
> **Working Title Candidates**:
> * *Teaching an AI to Read My Own Handwriting (Without Hallucinating My Thoughts)*
> * *500 Journals, 43,000 Lines, and the "Tuna vs. Turkey" Problem*
> * *From Vector Ink to Searchable Markdown: Fine-Tuning TrOCR on Apple Silicon*
> * *Why I Couldn't Just Slap an LLM on My Daily Journals*

---

## The Hook: Four Years Trapped in Digital Amber

* **The Stash**: Over the past four years, I've filled roughly 523 daily journal volumes on an Onyx BOOX e-ink tablet. They sit in my Obsidian vault as PDFs. Hundreds of megabytes of daily thoughts, code snippets, project frustrations, and weeknote fragments.
* **The Problem**: They are completely unsearchable. Unless I remember the exact Tuesday I wrote something down, it effectively doesn't exist. They're trapped in digital amber.
* **The Hand**: I write in an architectural drafting small-caps print. To human eyes, it looks neat, clean, and legible. To off-the-shelf OCR (Tesseract, Apple Vision, Google Cloud Vision, zero-shot TrOCR), it is pure, unadulterated gibberish:
  * Traditional OCR engines rely heavily on ascenders (`b, d, h, k`) and descenders (`g, j, p, q, y`) to calculate text baselines, x-height, and word gaps. Small caps strip those anchors away, causing baseline detectors to fracture words into nonsense.
  * In modern transformer vision-language models, the vision encoder recognizes capital glyphs (`W, E, R, E`) while the pretrained autoregressive language decoder expects lowercase tokens mid-word (`were`). The language prior battles the visual encoder, producing bizarre casing stutter and hallucinated substitutions.
* **The Stakes**: I want my journals in Obsidian as plain Markdown—indexed, linked, and searchable. But personal journals are the most intimate prose a human produces. I am not uploading 500 volumes of daily angst and unfiltered drafts to an unvetted commercial cloud API. It has to run on my own hardware.

---

## Act I: Peeking Under the PDF Hood (The Vector Stroke Surprise)

* **The Dot Grid Dread**: Both my BOOX tablet and my physical paper notebooks use a 5mm dot grid. Going into this project, I braced myself for weeks of OpenCV misery: morphological opening kernels, threshold tuning, and horizontal projection profiles trying to scrub out dot grids without deleting punctuation or the crossbars of my letters.
* **The Discovery**:
  * One evening I loaded a BOOX PDF export into Python (`pypdf`) to inspect the object streams.
  * What Onyx actually does under the hood is brilliant:
    * The dot-grid paper template is stored as a single, static background raster image (`/XObject /FXX1...`).
    * The handwriting is **not in the image at all**.
    * Every pen stroke is stored as native vector PDF `/Ink` annotations under the page's `/Annots` catalog.
  * Hundreds of raw coordinate paths with sub-pixel $(x, y)$ coordinate points, exact bounding boxes (`/Rect`), stylus width (`originStrokeWidth ~ 5.9pt`), and pressure dynamics.
* **The Relief**: The dot-grid problem vanished before it started. If you simply ignore the background raster image and render only the vector `/InkList` paths onto a blank white canvas, you get 100% pure digital ink on pure white. No noise, no paper texture, no morphological artifacts. Onyx handed me clean ground truth.

---

## Act II: Line Slicing Without Computer Vision (Geometric Clustering)

* **Why Computer Vision Line Detectors Failed**: Running YOLO or CRAFT line detectors on handwritten pages is notoriously brittle when baselines wander or punctuation floats.
* **The Geometric Pivot**: Because every stroke already had exact sub-pixel bounding coordinates $[x_1, y_1, x_2, y_2]$, line segmentation wasn't a computer vision problem—it was a 1D spatial clustering problem.
* **The Traps Along the Way**:
  * *The Floating Quote Trap*: Naive Y-center clustering pulled high quotation marks and apostrophes into the line above because their vertical center was closer to the previous line's baseline.
  * *The Two-Pass Filter Trap*: I tried classifying small marks ($<28\text{px}$) as boundary marks with custom geometry rules. It catastrophically broke the letters $H$, $A$, and $E$ by classifying their internal horizontal crossbars as stray punctuation and ripping them out of the letters.
  * *The Stem-Binding Solution*: Unified vertical proximity clustering (42px tolerance) paired with a simple geometric anchor: if a small horizontal stroke sits directly atop a vertical stem within a tight $\Delta X$ window, bind it to that line cluster. That cleanly anchored the crossbars of pronoun capital "I"s and $T$s.
  * *The 2022 Landscape Gutter*: Discovered that my early 2022 notebooks were written in landscape two-column mode ($2480 \times 1860\text{px}$). Slicing strokes across the 50px central gutter at $X=1240$ restored human reading order (left column top-to-bottom, then right column top-to-bottom).

---

## Act III: The Local Training Forge on Apple Silicon (84 to 2,721 Lines)

* **The Setup**:
  * Base model: `microsoft/trocr-base-handwritten` (~334M parameters: Vision Transformer encoder + BART decoder).
  * Hardware: MacBook Pro M5 Max (`portia`) using PyTorch Metal Performance Shaders (`mps`). Running at ~1.5 iterations/sec with zero cloud bills.
* **The Bootstrapping Flywheel**:
  * I built a zero-dependency local web UI (`label_ui.py`) using Python's standard library `http.server`. Keyboard-driven: `Enter` saves and advances, `Cmd+Left`/`Right` navigates.
  * **Batch 1 (84 lines)**: 100% manual transcription. Took ~15 minutes of typing. Fine-tuned 10 epochs. Result: **16.0% Character Error Rate (CER)**. The model was rough, but it had learned that my small-caps letters were lowercase text.
  * **Batch 2 (169 lines)**: I had Model 1 pre-transcribe the next volume. Instead of typing from scratch, I was just correcting a few typos. Took 5 minutes. Retrained: **6.44% CER** (>60% error reduction in one step!).
  * **Batches 3–15 (1,277 lines, spanning 2023–2026)**: Hit **2.50% CER**.
  * **Wave 3 Sprint (2,721 lines across 30 volumes)**: Hit **2.27% CER** with an evaluation loss of `0.2400` across 409 unseen validation lines. Character accuracy reached **>97.7%**.
* **The "Moving Exam" Dynamic**: An interesting ML phenomenon: whenever I added a new batch from an older year, validation error temporarily ticked *up*. The model was learning the evolution of my handwriting across time—stylus nib wear, changes in writing speed, and subtle shifts in small-caps spacing between 2022 and 2026.

---

## Act IV: The Hallucinatory Smoothing Trap & The "Tuna vs. Turkey" Moment

* **The Tempting Shortcut**: "Why bother fine-tuning a vision model when modern LLMs are so good? Why not just dump raw OCR into Claude or GPT-4 with a prompt like *'fix the typos in this transcript'?"*
* **The Experiment**: I tried it. The result was horrifying.
* **The "Tuna vs. Turkey" Smoking Gun**:
  * TrOCR transcribed a line about my cat:  
    `high spirits, begging for some time, from my lunch.`
  * The actual ink on the page was:  
    `high spirits, begging for some turkey, from my lunch.`
  * The LLM assistant smoothly rendered it as:  
    `"begging for some tuna from my lunch..."`
  * *Why?* Because Catsby is a cat, and in the LLM's pretraining web corpus, cats eat tuna. The language model subconsciously "corrected" corrupt tokens into internet stereotypes.
* **Identity Theft via Autocomplete**:
  * It replaced my authentic emotional phrasing **`awhile frettingly`** with the bland, generic cliché *"all the spring"*.
  * It hallucinated my former employer **`Mozilla`** into *"Movable"* (Movable Type).
  * It transformed my therapist **`Michael Crone`** into *"Michael Crowe"*.
  * It flattened idiosyncratic line starters like **`Still, though:`** into *"Sorting through,"*.
* **The Rule**: In personal journaling, **authorial voice is sacred**. A transcription that replaces your authentic vocabulary with high-probability corporate smoothing is worse than an OCR typo—it is literary forgery.

---

## Act V: Bounded Lexicon Constraints & The Levenshtein Clamp

* **The Vision Ceiling**: Why couldn't vision alone solve the last 2%? Because in rapid architectural drafting, a small-caps `G` vs `C`, an unclosed `D` loop vs `O`, or `find` vs `fine` can be physically identical down to a single sub-pixel ink stroke. Human readers use context; the model needed linguistic grounding without generative freedom.
* **Designing `clean_transcript.py`**:
  1. **The Hard Levenshtein Clamp ($\le 2$)**: Edits are mathematically bounded to a maximum edit distance of 2 (and $\le 1$ for short words). Words cannot expand into different phrases. `awhile frettingly` is programmatically shielded from becoming *"all the spring"*.
  2. **The 166,000-Word Obsidian Vault Lexicon**: Instead of relying on generic web dictionaries, I ingested unigram and bigram frequency distributions directly from 10 years of personal Obsidian notes (`journals/`, `pages/`, `reading/`). Pet names (*Catsby*, *Cheddars*), collaborators (*Dan Hon*), homelab jargon (*OPNsense*, *Proxmox*, *Tailscale*), and hardware (*Lamy*, *AL-star*, *BOOX*) are natively recognized as ground truth.
  3. **Morphological Lemmatization**: Suffix stripping (`-ed`, `-ing`, `-s`, `-ly`) ensures inflected English words (*textured*, *lunches*) aren't falsely flagged against dictionary base lemmas.
  4. **Frontmatter Immunity**: YAML fences (`---`) are locked down against modification.
* **The Result**: Applied 3,806 targeted word fixes across 509 files (~7.3 minor corrections per volume). Safely corrected `think table` $\to$ `e-ink tablet`, `to next` $\to$ `to text`, and `light is find` $\to$ `light is fine` without inventing a single syllable.

---

## Act VI: Sweeping the Archive: 43,199 Lines in 5 Hours

* **The Sweep**: Deployed `transcribe_journal.py` across all 523 BOOX journal PDFs.
* **Throughput**:
  * **522 of 523 volumes succeeded** (99.8% success rate) in **310 minutes (~5.1 hours)** on Portia's M5 Max (~15.8 seconds per volume). Exactly 1 error isolated to a truncated PDF stream.
  * **43,199 lines of handwriting** converted directly into Obsidian Markdown with frontmatter links back to the source PDFs.
  * Polished across the archive with the vault lexicon post-processor.
* **The Transformation**: After four years of opacity, 43,000 lines of handwriting became instantaneously searchable in Obsidian. Pressing `Cmd+Shift+F` and finding an entry from October 2022 in 20 milliseconds felt like magic.

---

## Act VII: The Zero-Friction Flywheel & The Homelab Split

* **The Reality of Maintenance**: A big batch sweep is great, but what happens tomorrow morning when I write another entry? Nobody wants to open a dedicated labeling UI every day.
* **The Asymmetric Homelab Split**:
  * **Linux Ingest VM on Proxmox (Always-on, CPU-only)**: Handles the daily churn. When a new PDF syncs from the tablet, it runs lightweight CPU inference (~60s), applies the vault lexicon polish, and writes the note to Obsidian.
  * **Portia (MacBook Pro M5 Max, Transient Laptop)**: The ML forge. When docked, it runs 90-second GPU bursts on accumulated corrections.
* **Zero-Friction Active Learning (`make harvest`)**:
  * You just read your daily journal in Obsidian like a normal person.
  * When you spot a typo in Markdown, you fix it.
  * Running `make harvest` automatically diffs your edited note against the raw baseline, crops the exact physical ink stroke strips from the source PDF, appends them to the dataset, commits, and pushes to a private homelab Forgejo repository (`0xDECAFBAD hub`).
  * No labeling UI. No manual bounding box drawing. Just reading and typing.
  * When the buffer hits 50–100 lines, `make simmer` bakes those corrections into the model weights with an experience replay buffer to prevent catastrophic forgetting.

---

## Epilogue: What Phase 1 Proved (and the Paper Horizon)

* **Key Takeaways for Homelab ML**:
  1. **The Single-Author Superpower**: Fine-tuning a compact 334M foundation model on *one human's habits* converges faster and achieves higher real-world fidelity than an unconstrained multi-billion-parameter cloud model.
  2. **Data Sovereignty Works**: 43,000 lines of intimate personal journals transcribed and refined locally with zero cloud API fees, zero surveillance, and complete privacy.
  3. **The Winning Architecture**: **Vision for Physical Ground Truth + Bounded Symbolic Lexicon Constraints for Disambiguation**.
* **Phase 2: The Physical Paper Frontier**:
  * Digital e-ink was the easy mode: no camera lens distortion, no uneven desk lighting, no paper curl, and vector strokes served up on a silver platter.
  * Phase 2 turns to physical ink-on-paper notebooks: morphological dot-grid removal from real paper scans, perspective rectification, shadow correction, and synthetic data generation.
  * But for digital ink? Phase 1 is done, working, and running daily.

---

## Visuals to Include

* **The Smoking Gun**: Side-by-side comparison of the "Tuna vs. Turkey" line: raw ink strip vs. TrOCR output vs. LLM hallucinatory smooth.
* **Vector Ink Decoupling**: Visual showing the PDF raster dot-grid background separated from the vector `/Ink` overlay.
* **Publication Charts**:
  * `charts/cer_by_epoch.png`: The steep progression from 16% to 2.27%.
  * `charts/scaling_law.png`: Empirical error rate plotted against training lines (log scale).
* **Obsidian Search in Action**: Animated GIF or screenshot searching across a 2022 handwritten passage in Obsidian.
