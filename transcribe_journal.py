#!/usr/bin/env python3
"""
End-to-End Journal Transcription CLI.

Takes any BOOX journal PDF, segments vector ink lines, runs batched TrOCR inference
via Apple Silicon MPS, detects paragraph breaks/dividers, and outputs clean Markdown.

Usage:
    uv run --with-requirements requirements.txt python3 transcribe_journal.py <path/to/journal.pdf>
    uv run --with-requirements requirements.txt python3 transcribe_journal.py <path/to/journal.pdf> --output journal.md
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
import pypdf
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Import segmentation and rendering mechanics from extract_lines
from extract_lines import (
    LineGroup,
    Stroke,
    cluster_strokes_into_lines,
    extract_page_strokes,
    render_line_strip,
)

try:
    from config import VAULT_DIR, MODEL_DIR
except ImportError:
    VAULT_DIR = Path.home() / "Documents/obsidian/main"
    MODEL_DIR = Path("trocr_finetuned/best_model")

from clean_transcript import LexiconModel, clean_markdown_document


def clean_ocr_text(text: str) -> str:
    """Normalize common OCR spacing artifacts around punctuation and contractions."""
    if not text:
        return ""
    # Detached contractions: "I 'm" -> "I'm", "It 's" -> "It's"
    text = re.sub(r"\b([A-Za-z]+)\s+(['’][A-Za-z]+)\b", r"\1\2", text)
    # Space before punctuation: "word ," -> "word,"
    text = re.sub(r"\s+([,.:;?!])", r"\1", text)
    # Quotes spacing: '" text "' -> '"text"'
    text = re.sub(r'([“"\'‘])\s+', r"\1", text)
    text = re.sub(r'\s+([”"\'’])', r"\1", text)
    # Degree symbol normalization (masculine ordinal º U+00BA -> degree ° U+00B0)
    text = text.replace("º", "°")
    # Repeated word stutters: "word word word" -> "word"
    text = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", text)
    # Double spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


class JournalTranscriber:
    def __init__(
        self,
        model_path: str = "trocr_finetuned/best_model",
        device: Optional[str] = None,
        batch_size: int = 8,
    ):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model_path = model_path
        print(f"Loading TrOCR model from {model_path} onto {self.device}...")
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.batch_size = batch_size

    def transcribe_lines(self, line_images: List[Image.Image]) -> List[str]:
        """Run batched TrOCR inference across a list of in-memory PIL Images."""
        predictions = []
        total = len(line_images)

        for i in range(0, total, self.batch_size):
            batch = line_images[i : i + self.batch_size]
            pixel_values = self.processor(batch, return_tensors="pt").pixel_values.to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=128,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.12,
                )

            decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            for text in decoded:
                predictions.append(clean_ocr_text(text))

        return predictions

    def process_page(
        self,
        page: pypdf.PageObject,
        page_num: int,
        save_crops_dir: Optional[Path] = None,
    ) -> Tuple[List[str], List[LineGroup], List[Stroke]]:
        """Extract strokes, cluster into lines, and transcribe a single page."""
        width, height, strokes = extract_page_strokes(page)
        lines = cluster_strokes_into_lines(strokes, page_number=page_num, page_width=width, page_height=height)
        rules = [s for s in strokes if s.is_rule]

        if not lines:
            return [], [], rules

        # Render in-memory line crops
        line_images = []
        for line in lines:
            img = render_line_strip(
                line,
                padding=18,
                padding_top=24,
                padding_bottom=28,
            )
            line_images.append(img)

            if save_crops_dir:
                crop_path = save_crops_dir / f"p{page_num:02d}_line{line.line_number:02d}.png"
                img.save(crop_path)

        texts = self.transcribe_lines(line_images)
        return texts, lines, rules


def format_markdown(
    page_results: List[Tuple[int, List[str], List[LineGroup], List[Stroke]]],
    source_pdf: Path,
    model_name: str,
) -> str:
    """Format transcribed lines into Markdown with paragraph breaks and dividers."""
    md_blocks: List[str] = []

    # Frontmatter
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", source_pdf.stem)
    journal_date = date_match.group(0) if date_match else source_pdf.stem

    frontmatter = [
        "---",
        f"date: {journal_date}",
        f'source: "[[notepads/journals/{source_pdf.name}]]"',
        f"source_pdf: {source_pdf.name}",
        f"transcribed_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"model: {model_name}",
        "---",
        "",
    ]
    md_blocks.extend(frontmatter)

    total_lines = 0

    for page_num, texts, lines, rules in page_results:
        if not lines:
            continue

        if len(page_results) > 1 and page_num > 1:
            md_blocks.append(f"\n<!-- Page {page_num} -->\n")

        # Combine lines into paragraphs by checking vertical spacing
        curr_paragraph: List[str] = []

        for idx, (text, line) in enumerate(zip(texts, lines)):
            total_lines += 1

            if not curr_paragraph:
                curr_paragraph.append(text)
            else:
                prev_line = lines[idx - 1]
                # Check baseline gap: standard line pitch is ~71px.
                # A gap > 115px indicates a blank line between paragraphs.
                baseline_gap = line.baseline_y - prev_line.baseline_y
                vertical_gap = line.bounding_box[1] - prev_line.bounding_box[3]

                # Check if a horizontal dividing rule sits between the two lines
                has_rule_between = any(
                    (prev_line.baseline_y < r.y_center < line.baseline_y) for r in rules
                )

                if has_rule_between:
                    md_blocks.append(" ".join(curr_paragraph))
                    md_blocks.append("\n---\n")
                    curr_paragraph = [text]
                elif baseline_gap > 115.0 or vertical_gap > 40.0:
                    # Paragraph break
                    md_blocks.append(" ".join(curr_paragraph))
                    md_blocks.append("")  # Blank line
                    curr_paragraph = [text]
                else:
                    curr_paragraph.append(text)

        if curr_paragraph:
            md_blocks.append(" ".join(curr_paragraph))

    return "\n".join(md_blocks).strip() + "\n"


def transcribe_single_pdf(
    transcriber: JournalTranscriber,
    pdf_file: Path,
    out_file: Path,
    pages: Optional[List[int]] = None,
    save_crops: bool = False,
    quiet: bool = False,
    lexicon: Optional[LexiconModel] = None,
) -> Tuple[int, float]:
    """Transcribe a single PDF file using an already initialized JournalTranscriber instance."""
    crops_dir = None
    if save_crops:
        crops_dir = out_file.parent / f"{pdf_file.stem}_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    reader = pypdf.PdfReader(str(pdf_file))
    num_pages = len(reader.pages)
    target_pages = pages if pages else list(range(1, num_pages + 1))

    page_results = []
    total_lines = 0

    for p_num in target_pages:
        page_idx = p_num - 1
        if page_idx < 0 or page_idx >= num_pages:
            continue
        p_t0 = time.time()
        texts, lines, rules = transcriber.process_page(reader.pages[page_idx], p_num, save_crops_dir=crops_dir)
        page_results.append((p_num, texts, lines, rules))
        total_lines += len(lines)
        p_elapsed = time.time() - p_t0

        if not quiet:
            print(f"    Page {p_num}: {len(lines)} lines in {p_elapsed:.1f}s")
            for l_idx, t in enumerate(texts[:2]):
                print(f"      L{l_idx+1:02d}: {repr(t[:50])}")
            if len(texts) > 2:
                print(f"      ... (+{len(texts)-2} more lines)")

    md_content = format_markdown(page_results, pdf_file, model_name=Path(transcriber.model_path).name)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    if lexicon is not None:
        raw_file = out_file.with_name(f"{out_file.stem}.raw.md")
        with open(raw_file, "w", encoding="utf-8") as rf:
            rf.write(md_content)
        cleaned_text, changes = clean_markdown_document(raw_file, lexicon)
        with open(out_file, "w", encoding="utf-8") as cf:
            cf.write(cleaned_text)
        if not quiet:
            print(f"    ✨ Polished with vault lexicon ({len(changes)} corrections). Raw baseline saved to: {raw_file.name}")
    else:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(md_content)

    elapsed = time.time() - t0
    return total_lines, elapsed


def transcribe_journal(
    pdf_path: str,
    output_path: Optional[str] = None,
    model_path: str = "trocr_finetuned/best_model",
    pages: Optional[List[int]] = None,
    batch_size: int = 8,
    save_crops: bool = False,
    quiet: bool = False,
) -> Path:
    """Convenience single-file entrypoint."""
    pdf_file = Path(pdf_path).resolve()
    if not pdf_file.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    if output_path:
        out_file = Path(output_path).resolve()
    else:
        out_dir = Path("transcribed")
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / f"{pdf_file.stem}.md"

    transcriber = JournalTranscriber(model_path=model_path, batch_size=batch_size)
    print(f"\nProcessing {pdf_file.name}...")
    lines, elapsed = transcribe_single_pdf(
        transcriber=transcriber,
        pdf_file=pdf_file,
        out_file=out_file,
        pages=pages,
        save_crops=save_crops,
        quiet=quiet,
    )
    print(f"Completed {pdf_file.name}: {lines} lines in {elapsed:.1f}s -> {out_file}\n")
    return out_file


def resolve_pdf_paths(paths_or_globs: List[str]) -> List[Path]:
    """Resolve a list of paths and glob patterns into sorted, unique PDF files."""
    matched = []
    for p in paths_or_globs:
        expanded = os.path.expanduser(p)
        candidates = [Path(expanded)]
        if VAULT_DIR.exists():
            candidates.append(VAULT_DIR / "notepads/journals" / p)
            candidates.append(VAULT_DIR / "notepads/journals" / expanded)

        found_candidate = False
        for c in candidates:
            if c.is_file() and c.suffix.lower() == ".pdf":
                matched.append(c.resolve())
                found_candidate = True
                break
            elif c.is_dir():
                matched.extend(sorted(c.glob("*.pdf")))
                matched.extend(sorted(c.glob("*.PDF")))
                found_candidate = True
                break

        if not found_candidate:
            if any(char in expanded for char in "*?[]"):
                files = glob.glob(expanded)
                if not files and VAULT_DIR.exists():
                    files = glob.glob(str(VAULT_DIR / "notepads/journals" / expanded))
                matched.extend([Path(f).resolve() for f in files if f.lower().endswith(".pdf")])
            else:
                print(f"Warning: '{p}' is not a PDF file or matching directory, skipping.")

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for f in matched:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return sorted(unique)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end transcription of BOOX journal PDFs to Markdown (single file or batch/glob)."
    )
    parser.add_argument("pdf_paths", nargs="*", help="Path(s), directory, or glob pattern(s) of BOOX journal PDFs (default: scan vault for new PDFs)")
    default_out = str(VAULT_DIR / "notepads/transcriptions") if VAULT_DIR.exists() else "transcribed"
    parser.add_argument("-o", "--output", default=default_out, help=f"Output Markdown file (single input) or directory (batch) (default: {default_out})")
    parser.add_argument("--model", default=str(MODEL_DIR), help=f"Path to TrOCR model checkpoint (default: {MODEL_DIR})")
    parser.add_argument("--pages", nargs="+", type=int, help="Specific 1-based page numbers (default: all)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference (default: 8)")
    parser.add_argument("--save-crops", action="store_true", help="Save intermediate line crop images for review")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if target Markdown exists and is non-empty")
    parser.add_argument("--by-year", action="store_true", default=True, help="Nest outputs into YYYY/ subdirectories based on filename (default: True)")
    parser.add_argument("--no-by-year", dest="by_year", action="store_false", help="Disable YYYY/ subdirectories")
    parser.add_argument("--clean", action="store_true", default=True, help="Apply Levenshtein vault lexicon cleanup and save .raw.md baseline (default: True)")
    parser.add_argument("--no-clean", dest="clean", action="store_false", help="Disable automatic lexicon cleanup")
    parser.add_argument("--manifest", help="Path to write JSON manifest summary (default: <output_dir>/manifest.json)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress line-by-line verbose output")

    args = parser.parse_args()

    if not args.pdf_paths:
        if VAULT_DIR.exists():
            args.pdf_paths = [str(VAULT_DIR / "notepads/journals/*.pdf")]
            args.skip_existing = True
        else:
            print("Error: No PDF files specified and VAULT_DIR not found.")
            sys.exit(1)

    files = resolve_pdf_paths(args.pdf_paths)

    if args.skip_existing and args.output:
        out_base = Path(args.output).resolve()
        untranscribed = []
        for pdf in files:
            year_match = re.search(r"(\d{4})-\d{2}-\d{2}", pdf.stem)
            year_folder = year_match.group(1) if year_match else "misc"
            target_md = (out_base / year_folder / f"{pdf.stem}.md") if args.by_year else (out_base / f"{pdf.stem}.md")
            if not (target_md.exists() and target_md.stat().st_size > 50):
                untranscribed.append(pdf)
        files = untranscribed

    if not files:
        print("✨ All journals up to date! (0 untranscribed PDFs found in notepads/journals/).")
        return

    print(f"Found {len(files)} new PDF volume(s) to transcribe.")

    lexicon = None
    if args.clean:
        if not args.quiet:
            print("Loading vault vocabulary lexicon for bounded post-processing...")
        lexicon = LexiconModel()
        lexicon.load_or_build()

    # Single-file target with custom filename
    is_single_file_custom = (
        len(files) == 1
        and args.output
        and not Path(args.output).is_dir()
        and (args.output.endswith(".md") or not Path(args.output).suffix)
    )

    if is_single_file_custom:
        out_file = Path(args.output).resolve()
        transcriber = JournalTranscriber(model_path=args.model, batch_size=args.batch_size)
        transcribe_single_pdf(
            transcriber=transcriber,
            pdf_file=files[0],
            out_file=out_file,
            pages=args.pages,
            save_crops=args.save_crops,
            quiet=args.quiet,
            lexicon=lexicon,
        )
        print(f"\n✨ Complete! Markdown saved to: {out_file}")
        return

    # Batch output directory
    out_dir = Path(args.output).resolve() if args.output else Path("transcribed").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest).resolve() if args.manifest else (out_dir / "manifest.json")
    manifest_data = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception:
            manifest_data = {}

    print(f"Output directory: {out_dir}")
    print(f"Manifest path:    {manifest_path}")
    if args.skip_existing:
        print("Flag active:      --skip-existing enabled")
    if args.by_year:
        print("Flag active:      --by-year subfolder routing enabled")

    transcriber = JournalTranscriber(model_path=args.model, batch_size=args.batch_size)

    total_batch_lines = 0
    total_batch_time = 0.0
    files_processed = 0
    files_skipped = 0
    files_errored = 0
    recent_durations: List[float] = []

    batch_t0 = time.time()

    for idx, pdf_file in enumerate(files, 1):
        # Route to year subfolder if requested
        if args.by_year:
            year_match = re.search(r"(\d{4})-\d{2}-\d{2}", pdf_file.stem)
            year_folder = year_match.group(1) if year_match else "misc"
            target_dir = out_dir / year_folder
        else:
            target_dir = out_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        target_md = target_dir / f"{pdf_file.stem}.md"

        # Check skip condition
        if args.skip_existing and target_md.exists() and target_md.stat().st_size > 50:
            if not args.quiet:
                print(f"[{idx}/{len(files)}] ⏭️  Skipping {pdf_file.name} (already exists: {target_md.name})")
            files_skipped += 1
            if pdf_file.name not in manifest_data:
                manifest_data[pdf_file.name] = {
                    "status": "skipped",
                    "out_file": str(target_md),
                    "timestamp": datetime.now().isoformat(),
                }
            continue

        pct = (idx / len(files)) * 100
        print(f"\n[{idx}/{len(files)}] ({pct:.1f}%) Transcribing {pdf_file.name}...")

        try:
            lines, elapsed = transcribe_single_pdf(
                transcriber=transcriber,
                pdf_file=pdf_file,
                out_file=target_md,
                pages=args.pages,
                save_crops=args.save_crops,
                quiet=args.quiet,
                lexicon=lexicon,
            )
            total_batch_lines += lines
            total_batch_time += elapsed
            files_processed += 1
            recent_durations.append(elapsed)
            if len(recent_durations) > 20:
                recent_durations.pop(0)

            # ETA calculation based on recent average
            avg_speed = sum(recent_durations) / len(recent_durations)
            rem_files = len(files) - idx
            eta_sec = rem_files * avg_speed
            eta_m = int(eta_sec // 60)
            eta_s = int(eta_sec % 60)

            manifest_data[pdf_file.name] = {
                "status": "ok",
                "lines": lines,
                "duration_sec": round(elapsed, 2),
                "out_file": str(target_md),
                "timestamp": datetime.now().isoformat(),
            }

            print(f"  -> {lines} lines in {elapsed:.1f}s -> {target_md.name} | ETA: {eta_m}m {eta_s}s")

        except Exception as e:
            files_errored += 1
            err_msg = f"{type(e).__name__}: {e}"
            print(f"  ❌ Error processing {pdf_file.name}: {err_msg}")
            manifest_data[pdf_file.name] = {
                "status": "error",
                "error": err_msg,
                "timestamp": datetime.now().isoformat(),
            }
            with open(out_dir / "transcription_errors.log", "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat()}] {pdf_file.name}: {err_msg}\n")

        # Periodically flush manifest atomically
        try:
            tmp_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
            with open(tmp_manifest, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)
            tmp_manifest.replace(manifest_path)
        except Exception as e:
            with open(out_dir / "transcription_errors.log", "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat()}] Manifest write error: {e}\n")

    total_wall_time = time.time() - batch_t0
    mins = int(total_wall_time // 60)
    secs = int(total_wall_time % 60)
    avg_time = (total_batch_time / files_processed) if files_processed else 0

    print("\n" + "=" * 65)
    print(f"✨ Sweep Finished across {len(files)} volumes.")
    print(f"   Processed: {files_processed}  |  Skipped: {files_skipped}  |  Errors: {files_errored}")
    print(f"📊 Total lines transcribed: {total_batch_lines}")
    print(f"⏱️  Total wall time:         {mins}m {secs}s (avg {avg_time:.1f}s/vol)")
    print(f"📁 Transcriptions saved to:  {out_dir}")
    print(f"📋 Manifest saved to:        {manifest_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
