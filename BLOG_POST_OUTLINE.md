# Blog Post Outline: Training a Local Vision Model to Read My Own Handwriting

> Working Title Ideas:
> - *Teaching an AI to Read My Own Handwriting (Without Hallucinating My Thoughts)*
> - *500 Journals, 43,000 Lines, and the Hallucinatory Smoothing Trap*
> - *Fine-Tuning TrOCR on Apple Silicon: From Vector Ink to Searchable Markdown*
> - *Why I Couldn't Just Slap an LLM on My Scanned Journals*

---

## The Hook: 500 Journals Trapped in Digital Ink

* **The Setup**: Over the past four years, I've filled ~500 daily journal notebooks on an Onyx BOOX e-ink tablet. They sit in my Obsidian vault as PDFs.
* **The Lettering Style**: Architectural drafting small-caps. To human eyes, it looks neat, clean, and legible. To off-the-shelf OCR (Tesseract, cloud vision APIs, zero-shot TrOCR), it is complete gibberish:
  * Small-caps characters lack standard lowercase ascenders (`b, d, h`) and descenders (`p, q, y`), confusing baseline calculators.
  * In transformer vision-language models, the vision encoder recognizes uppercase glyphs (`W, E, R, E`) while the pretrained autoregressive language decoder expects lowercase tokens mid-word (`were`), sparking internal decoder conflict and character hallucination.
* **The Dilemma**: I want my daily journals indexed, searchable, and linkable in Obsidian. Transcribing them by hand would take months; commercial handwriting cloud APIs violate personal privacy; and generic models butcher the text.

---

## Act I: Peeking Under the PDF Hood (The Vector Stroke Discovery)

* **The Dot Grid Dread**: Both my digital tablet and physical notebooks use a dot grid. I went into this assuming I'd spend weeks writing OpenCV morphological filters, erosion kernels, and horizontal projection profiles to slice out dot grids without deleting letter stems.
* **Peeking Under the PDF Hood**:
  * Opening the BOOX PDF export in Python (`pypdf`) revealed an incredible architecture choice by Onyx:
  * The dot grid is a static background raster image (`/XObject`).
  * The handwriting is **not in the image at all**.
  * The pen strokes are stored as native vector PDF `/Ink` annotations under `/Annots`—hundreds of raw coordinate paths with sub-pixel $(x, y)$ points, stroke widths (`5.9 pt`), pressure dynamics, and timestamps.
* **The Epiphany**: By simply ignoring the background image and rendering only the vector ink onto a blank white canvas, the dot-grid problem vanished entirely. Pristine, anti-aliased digital ink on pure white.

---

## Act II: Line Slicing Without OpenCV (Geometric Clustering & Stem-Binding)

* **Why Traditional Line Segmentation Breaks**: CRAFT, YOLO bounding boxes, Voronoi tessellations, and projection profiles frequently break when baselines wander or characters lack ascenders/descenders.
* **Geometric Clustering in 1D**: Because every pen stroke already has exact bounding coordinates $[x_1, y_1, x_2, y_2]$, grouping strokes into lines became a spatial clustering problem.
* **The Subtle Edge Cases**:
  * *The Two-Pass Filter Trap*: Trying to classify small marks (crossbars, commas) with handcrafted boundary rules backfired—it stripped internal crossbars from $H$, $A$, and $E$.
  * *Stem-Binding*: Floating top bars on tall pronoun "I"s and "T"s were solved with a simple geometric anchor: if a horizontal mark sits directly atop a vertical stem within a tight $\Delta X$ window, it binds to that line cluster.
  * *The 2022 Landscape Gutter*: Discovered that early 2022 journals were landscape two-column pages. Slicing left-column top-to-bottom, then right-column top-to-bottom restored perfect human reading order.

---

## Act III: The Active Learning Flywheel on Apple Silicon (84 to 2,721 Lines)

* **Local Machine Learning on Apple Silicon**:
  * Architecture: `microsoft/trocr-base-handwritten` (~334M parameters: Vision Transformer encoder + BART decoder).
  * Hardware: Apple Silicon M5 Max MacBook Pro (`portia`) using PyTorch Metal Performance Shaders (`mps`).
  * Throughput: ~1.5 iterations/sec (~12 to 16 minutes per 10-epoch run).
* **The Zero-Dependency Annotation UI (`label_ui.py`)**:
  * Built a tiny, zero-dependency Python `http.server` with keyboard shortcuts (`Enter` to save/advance) and a multi-batch dropdown switcher across 30 batches.
* **The Empirical Scaling Trajectory**:
  * **Batch 1 (84 lines)**: 100% manual typing (~15 min) $\to$ **16.0% CER**.
  * **Batch 2 (169 lines)**: Pre-transcribed by Model 1 $\to$ mostly typo fixes (~5 min) $\to$ **6.44% CER** (>60% error reduction!).
  * **Batches 3 & 4 (334 lines)**: Pre-transcribed by Model 2 $\to$ ~3 min per batch $\to$ **6.33% CER**.
  * **Batches 5–15 (1,277 lines, spanning 2023–2026)**: Hit **2.50% CER** at Epoch 10.
  * **Wave 3 Doubling Sprint (2,721 lines across 30 volumes)**: Hit project-best **2.27% CER** (eval loss: `0.2400`, >97.7% character accuracy).
* **The "Moving Exam" Dynamic**: Why validation error temporarily ticked up as we added older years—the model was learning historical stylus wear and writing speed changes across a 5-year timeline.

---

## Act IV: The Hallucinatory Smoothing Trap & The "Tuna vs. Turkey" Moment

* **The Tempting Shortcut**: "Why fine-tune a specialized vision model when you can just pipe raw OCR through Claude or GPT-4 with a prompt like *'clean up this messy OCR'*?"
* **The Experiment & The Horror**:
  * Because a text-only LLM is blind to the physical ink strokes, it resolves ambiguous character strings purely by statistical probability in its pretraining corpus.
  * It erased authentic, idiosyncratic personal voice and substituted bland language-model clichés:
    * Replaced authentic emotional phrase **`awhile frettingly`** with *"all the spring"*.
    * Hallucinated employer **`Mozilla`** into *"Movable"* (Movable Type).
    * Hallucinated therapist **`Michael Crone`** into *"Michael Crowe"*.
    * Flattened **`Still, though:`** into *"Sorting through,"*.
* **The "Tuna vs. Turkey" Smoking Gun**:
  * TrOCR transcribed: `high spirits, begging for some time, from my lunch.`
  * Actual handwriting: `high spirits, begging for some turkey, from my lunch.`
  * The LLM assistant summarized it as: *"begging for some tuna from my lunch..."*
  * Because Catsby is a cat and cats eat tuna, the language model subconsciously smoothed the corrupt token into an internet stereotype.
* **The Invariant**: In personal journals, **voice is sacred**. A transcription that replaces your authentic vocabulary with statistical corporate smoothing is worse than an OCR typo—it is a forgery.

---

## Act V: The Last Mile: Bounded Lexicon Constraints & The Levenshtein Clamp

* **The Physical Ceiling of Vision Models**:
  * Why vision models alone plateau around 2.2% CER: In architectural drafting, a small-caps `G` vs `C`, an unclosed `d` loop vs `e`, or `find` vs `fine` can be physically identical down to a single sub-pixel stroke.
  * To get below 0.5% word error rate without LLM hallucination, we needed *linguistic grounding without generative freedom*.
* **Designing `clean_transcript.py`**:
  1. **The Hard Levenshtein Clamp ($\le 2$)**: Programmatically prohibits any edit with distance $>2$ (and $>1$ for short words). Phrasing like `awhile frettingly` is mathematically shielded from being rewritten into *"all the spring"*.
  2. **The 166k-Word Obsidian Vault Lexicon**: Ingested unigram and bigram frequency distributions directly from 10 years of personal Obsidian notes (`journals/`, `pages/`, `reading/`). Pet names (*Catsby*, *Cheddars*), collaborators (*Dan Hon*), and hardware (*Lamy*, *AL-star*, *BOOX*) are natively recognized as ground truth.
  3. **Morphological Lemmatization**: Implemented inflection stripping (`-ed`, `-ing`, `-s`, `-ly`) so standard inflected words (*textured*, *lunches*) aren't falsely flagged against base lemmas in `/usr/share/dict/words`.
  4. **Frontmatter Immunity**: YAML fences (`---`) are locked down against modification.
* **Results Across the Archive**:
  * Cleaned 509 files, applying 3,806 targeted word fixes (~7.3 minor corrections per volume).
  * Safely fixed `think table` $\to$ `e-ink tablet`, `to next` $\to$ `to text`, and `light is find` $\to$ `light is fine`.

---

## Act VI: Sweeping the Archive: 43,199 Lines in 5 Hours

* **Unattended Production Batch**:
  * Deployed `transcribe_journal.py` across all 523 BOOX journal PDFs.
  * **522 of 523 volumes succeeded** (99.8% success rate) in **310 minutes (~5.1 hours)** on Portia's M5 Max (~15.8s per volume). Exactly 1 error isolated to a truncated PDF stream.
  * **43,199 lines of handwriting** converted directly into Obsidian Markdown with frontmatter links back to the source PDFs.
* **Key Takeaways for Personal Machine Learning & Homelab AI**:
  1. **The Single-Author Superpower**: Fine-tuning a compact 334M foundation model on *one human's habits* converges faster and achieves higher real-world accuracy than multi-billion-parameter cloud models.
  2. **Local, Private, and Sovereign**: Intimate journal entries never left local silicon. Zero cloud fees, zero data leaks, zero subscription dependencies.
  3. **The Winning Architecture**: **Vision for Physical Ground Truth + Bounded Symbolic Lexicon Constraints for Disambiguation**.

---

## Visual Artifacts for the Post

* **`charts/cer_by_epoch.png`**: High-res CER curves showing the progression from 16% to 2.27%.
* **`charts/scaling_law.png`**: Empirical scaling law curve plotting validation error against lines of training data.
* **`charts/loss_by_epoch.png`**: Multi-run loss curves demonstrating the 10-epoch plateau.
* **Side-by-Side Images**: Raw ink strip $\leftrightarrow$ TrOCR transcription $\leftrightarrow$ Lexicon-cleaned text.
* **Interactive Canvas**: Canvas dashboard visualization (`canvas_14`).
