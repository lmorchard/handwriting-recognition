"""
harvest_corrections.py - Extract Ground-Truth Corrections from User-Edited Obsidian Notes.

Workflow:
1. Compares current Obsidian transcript against the raw unedited baseline (from backup tarball).
2. Extracts corresponding line stroke strips from the source PDF.
3. Aligns edited paragraph text back to the physical line crops.
4. Saves cropped PNGs and updates `spot_corrections/dataset.csv` for future simmer retraining.
"""

import argparse
import csv
import difflib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
import tarfile
from typing import Dict, List, Optional, Tuple

import pypdf
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Local project imports
from extract_lines import LineGroup, cluster_strokes_into_lines, extract_page_strokes, render_line_strip

try:
    from config import DATA_DIR, MODEL_DIR, BACKUP_TARBALL, VAULT_DIR
except ImportError:
    DATA_DIR = Path("../handwriting-data").resolve()
    MODEL_DIR = Path("trocr_finetuned/best_model").resolve()
    VAULT_DIR = Path.home() / "Documents/obsidian/main"
    BACKUP_TARBALL = VAULT_DIR / "notepads/transcriptions_backup_raw_20261022.tar.gz"


def find_raw_transcript_in_tarball(tar_path: Path, rel_path: str) -> Optional[str]:
    """Extract original unedited transcript text from the backup tarball."""
    if not tar_path.exists():
        return None
    with tarfile.open(tar_path, "r:gz") as tar:
        # Search candidate names inside tarball
        candidates = [
            f"transcriptions/{rel_path}",
            rel_path,
            Path(rel_path).name,
        ]
        for cand in candidates:
            try:
                f = tar.extractfile(cand)
                if f:
                    return f.read().decode("utf-8", errors="replace")
            except KeyError:
                pass
        # Fallback: search members by filename
        target_name = Path(rel_path).name
        for member in tar.getmembers():
            if member.name.endswith(target_name):
                f = tar.extractfile(member)
                if f:
                    return f.read().decode("utf-8", errors="replace")
    return None


def parse_markdown_paragraphs(text: str) -> List[Tuple[int, str]]:
    """Parse Markdown text into (page_number, paragraph_text) blocks."""
    lines = text.splitlines()
    in_frontmatter = False
    frontmatter_closed = False
    current_page = 1
    paragraphs = []
    curr_para = []

    for l_idx, line in enumerate(lines):
        stripped = line.strip()
        if not frontmatter_closed:
            if l_idx == 0 and stripped == "---":
                in_frontmatter = True
                continue
            elif in_frontmatter and stripped == "---":
                in_frontmatter = False
                frontmatter_closed = True
                continue
            elif in_frontmatter:
                continue

        # Page comment markers
        page_match = re.match(r"^<!--\s*Page\s+(\d+)\s*-->", stripped, re.IGNORECASE)
        if page_match:
            if curr_para:
                paragraphs.append((current_page, " ".join(curr_para).strip()))
                curr_para = []
            current_page = int(page_match.group(1))
            continue

        if stripped == "---":
            # Horizontal rule divider
            if curr_para:
                paragraphs.append((current_page, " ".join(curr_para).strip()))
                curr_para = []
            continue

        if not stripped:
            # Blank line paragraph break
            if curr_para:
                paragraphs.append((current_page, " ".join(curr_para).strip()))
                curr_para = []
            continue

        curr_para.append(stripped)

    if curr_para:
        paragraphs.append((current_page, " ".join(curr_para).strip()))

    return [(p, text) for p, text in paragraphs if text]


def align_and_slice_corrections(
    raw_lines: List[str],
    edited_para: str,
) -> List[Tuple[int, str, str]]:
    """
    Given a list of raw TrOCR lines that formed a paragraph and the user's edited paragraph,
    maps words back to each line strip.
    Returns: list of (line_idx, raw_line_text, corrected_line_text) for lines that changed.
    """
    raw_words_per_line = [re.findall(r"\S+", l) for l in raw_lines]
    flat_raw_words = []
    word_to_line_map = []

    for l_idx, words in enumerate(raw_words_per_line):
        for w in words:
            flat_raw_words.append(w)
            word_to_line_map.append(l_idx)

    edited_words = re.findall(r"\S+", edited_para)

    # Sequence alignment between words
    matcher = difflib.SequenceMatcher(None, flat_raw_words, edited_words)
    aligned_edited_words_per_line: Dict[int, List[str]] = {i: [] for i in range(len(raw_lines))}

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("equal", "replace"):
            # Words in edited range [j1:j2] correspond to raw range [i1:i2]
            if i2 > i1:
                # Distribute edited words across the involved lines
                for offset, j in enumerate(range(j1, j2)):
                    # Assign proportionally to the line index of the corresponding raw word
                    raw_idx = min(i1 + offset, i2 - 1)
                    l_idx = word_to_line_map[raw_idx]
                    aligned_edited_words_per_line[l_idx].append(edited_words[j])
            else:
                # Pure insertion: attach to preceding word's line
                l_idx = word_to_line_map[min(i1, len(word_to_line_map) - 1)] if word_to_line_map else 0
                for j in range(j1, j2):
                    aligned_edited_words_per_line[l_idx].append(edited_words[j])
        elif tag == "delete":
            # Words deleted from raw; nothing added to edited
            pass
        elif tag == "insert":
            # Pure insertion
            l_idx = word_to_line_map[min(i1, len(word_to_line_map) - 1)] if word_to_line_map else 0
            for j in range(j1, j2):
                aligned_edited_words_per_line[l_idx].append(edited_words[j])

    changes = []
    for l_idx in range(len(raw_lines)):
        orig_text = raw_lines[l_idx].strip()
        new_text = " ".join(aligned_edited_words_per_line[l_idx]).strip()
        if new_text and new_text != orig_text:
            changes.append((l_idx, orig_text, new_text))

    return changes


def harvest_corrections_from_note(
    note_path: Path,
    tar_backup_path: Path,
    out_dir: Path,
    model_path: str = "trocr_finetuned/best_model",
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
) -> List[Dict]:
    """Harvest corrections from a single edited Obsidian note."""
    print(f"\nAnalyzing note: {note_path.name}")
    with open(note_path, "r", encoding="utf-8") as f:
        edited_text = f.read()

    # Find source PDF
    source_match = re.search(r'source:\s*"\[\[(.*?)\]\]"', edited_text)
    if not source_match:
        source_match = re.search(r"source_pdf:\s*([^\s]+)", edited_text)
    if not source_match:
        print("  Error: Could not determine source PDF from frontmatter.")
        return []

    pdf_rel = source_match.group(1).replace("notepads/journals/", "")
    pdf_candidates = [
        VAULT_DIR / "notepads/journals" / pdf_rel,
        Path("/Users/lmorchard/Documents/obsidian/main/notepads/journals") / pdf_rel,
        Path.home() / ".decafclaw/decafclaw/obsidian/main/notepads/journals" / pdf_rel,
        Path("notepads/journals") / pdf_rel,
    ]
    pdf_path = next((p for p in pdf_candidates if p.exists()), None)
    if not pdf_path:
        print(f"  Error: Source PDF not found: {pdf_rel}")
        return []

    # Get raw baseline text: check for local <note>.raw.md first, then fallback to backup tarball
    raw_file = note_path.with_name(f"{note_path.stem}.raw.md")
    if raw_file.exists():
        print(f"  Found local raw baseline: {raw_file.name}")
        with open(raw_file, "r", encoding="utf-8") as rf:
            raw_text = rf.read()
    else:
        rel_note = f"{note_path.parent.name}/{note_path.name}"
        raw_text = find_raw_transcript_in_tarball(tar_backup_path, rel_note)
        if raw_text:
            print(f"  Found raw baseline in backup tarball for {rel_note}")

    if not raw_text:
        print(f"  Warning: Raw baseline for {note_path.name} not found (checked {raw_file.name} and backup tarball).")
        return []

    raw_paras = parse_markdown_paragraphs(raw_text)
    edited_paras = parse_markdown_paragraphs(edited_text)

    if len(raw_paras) != len(edited_paras):
        print(f"  Note has structural paragraph differences (raw: {len(raw_paras)}, edited: {len(edited_paras)}). Aligning by page...")

    # Load model and reader
    print(f"  Loading TrOCR model on {device}...")
    model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
    processor = TrOCRProcessor.from_pretrained(model_path)
    model.eval()

    reader = pypdf.PdfReader(str(pdf_path))
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    harvested_records = []

    # Process page by page
    for p_num in range(1, len(reader.pages) + 1):
        # Check if any paragraphs on this page changed
        p_raw_paras = [text for (pg, text) in raw_paras if pg == p_num]
        p_edited_paras = [text for (pg, text) in edited_paras if pg == p_num]

        if not p_edited_paras:
            continue
        if p_raw_paras == p_edited_paras:
            continue  # Page untouched

        print(f"  -> Page {p_num}: User corrections detected!")
        w, h, strokes = extract_page_strokes(reader.pages[p_num - 1])
        lines = cluster_strokes_into_lines(strokes, page_number=p_num, page_width=w, page_height=h)

        # Transcribe lines to get exact raw line boundaries
        line_images = [render_line_strip(l) for l in lines]
        raw_line_texts = []
        for img in line_images:
            pix = processor(images=[img], return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                out = model.generate(pix, max_length=128)
            raw_line_texts.append(processor.batch_decode(out, skip_special_tokens=True)[0].strip())

        # Align edited text for this page
        full_p_raw = " ".join(p_raw_paras)
        full_p_edited = " ".join(p_edited_paras)

        changes = align_and_slice_corrections(raw_line_texts, full_p_edited)
        print(f"     Extracted {len(changes)} corrected line(s) on Page {p_num}:")

        for l_idx, raw_l, corr_l in changes:
            # Save cropped PNG
            img_filename = f"{pdf_path.stem}_p{p_num:02d}_l{l_idx+1:02d}.png"
            img_dest = images_dir / img_filename
            line_images[l_idx].save(img_dest)

            diff_summary = []
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, raw_l.split(), corr_l.split()).get_opcodes():
                if tag != "equal":
                    diff_summary.append(f"{' '.join(raw_l.split()[i1:i2])} -> {' '.join(corr_l.split()[j1:j2])}")

            print(f"       Line {l_idx+1:02d}: {', '.join(diff_summary)}")
            harvested_records.append({
                "image_path": str(img_dest.relative_to(out_dir)),
                "ground_truth": corr_l,
                "raw_prediction": raw_l,
                "diff": ", ".join(diff_summary),
                "source_note": note_path.name,
                "page": p_num,
                "line": l_idx + 1,
            })

    return harvested_records


def main():
    parser = argparse.ArgumentParser(description="Harvest Ground-Truth Corrections from User-Edited Obsidian Notes.")
    parser.add_argument("note", help="Path to edited Markdown note (or directory of notes)")
    parser.add_argument(
        "--tarball",
        default=str(BACKUP_TARBALL),
        help=f"Path to raw backup tarball (default: {BACKUP_TARBALL})",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DATA_DIR / "spot_corrections"),
        help=f"Output directory for harvested dataset and images (default: {DATA_DIR / 'spot_corrections'})",
    )
    parser.add_argument("--model", default=str(MODEL_DIR), help=f"Path to TrOCR model checkpoint (default: {MODEL_DIR})")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = Path(args.tarball)

    target_path = Path(args.note)
    if target_path.is_file():
        files = [target_path]
    elif target_path.is_dir():
        files = sorted(target_path.rglob("*.md"))
    else:
        print(f"Error: Path not found: {target_path}")
        sys.exit(1)

    all_harvested = []
    for f in files:
        records = harvest_corrections_from_note(f, tar_path, out_dir, model_path=args.model)
        all_harvested.extend(records)

    if not all_harvested:
        print("\nNo corrections found or notes are identical to baseline.")
        return

    csv_path = out_dir / "dataset.csv"
    file_exists = csv_path.exists()
    keys = ["image_path", "ground_truth", "raw_prediction", "diff", "source_note", "page", "line"]

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists:
            writer.writeheader()
        for r in all_harvested:
            writer.writerow(r)

    print("\n" + "=" * 65)
    print(f"SUCCESS: Harvested {len(all_harvested)} corrected line(s) into {csv_path}!")
    print(f"Line images stored in: {out_dir / 'images'}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
