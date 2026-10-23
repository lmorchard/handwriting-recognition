"""
Stage Wave 3: 15 volumes for handwriting line verification.
Extracts line strips using extract_lines.py, then runs batched
pre-transcription using trocr_finetuned/best_model on Apple Silicon MPS.
"""

import csv
import json
import os
from pathlib import Path
import sys
import time
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import extract_lines
from transcribe_journal import clean_ocr_text

VOLUMES = [
    "2022-10-19.pdf",
    "2023-01-26.pdf",
    "2023-03-26.pdf",
    "2023-10-25.pdf",
    "2024-03-18.pdf",
    "2024-06-05.pdf",
    "2024-08-14.pdf",
    "2024-11-20.pdf",
    "2025-08-06.pdf",
    "2025-08-11.pdf",
    "2025-08-27.pdf",
    "2025-08-28.pdf",
    "2026-01-12.pdf",
    "2026-03-25.pdf",
    "2026-04-21.pdf",
]


def main():
    base_pdf_dir = Path("/Users/lmorchard/Documents/obsidian/main/notepads/journals")
    model_path = "trocr_finetuned/best_model"

    print("==================================================================")
    print(f"  Staging Wave 3: {len(VOLUMES)} volumes across 2022-2026")
    print("==================================================================")

    # 1. Line extraction pass
    print("\n[Phase 1] Extracting line strips from PDF vectors...")
    staged_batches = []
    total_lines = 0

    for idx, v_name in enumerate(VOLUMES, 1):
        pdf_path = base_pdf_dir / v_name
        if not pdf_path.exists():
            print(f"  ❌ Error: {pdf_path} not found!")
            continue

        stem = pdf_path.stem
        # e.g. batch_2025_08_28
        batch_name = f"batch_{stem.replace('-', '_')}"
        batch_dir = Path(batch_name)

        print(f"\n  [{idx}/{len(VOLUMES)}] Extracting {v_name} -> {batch_name}...")
        catalog = extract_lines.process_pdf(str(pdf_path), str(batch_dir))
        print(f"    -> Extracted {len(catalog)} lines.")
        total_lines += len(catalog)
        staged_batches.append((batch_dir, stem, catalog))

    print(f"\nPhase 1 Complete: {len(staged_batches)} batches, {total_lines} total lines.")

    # 2. Batched GPU pre-transcription pass
    print("\n[Phase 2] Loading fine-tuned TrOCR model onto MPS...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Device: {device}")
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
    model.eval()

    print(f"\n[Phase 3] Pre-transcribing all batches via GPU...")
    t_start = time.time()
    batch_size = 16

    for b_idx, (batch_dir, stem, catalog) in enumerate(staged_batches, 1):
        b_t0 = time.time()
        print(f"\n  [{b_idx}/{len(staged_batches)}] Pre-transcribing {batch_dir.name} ({len(catalog)} lines)...")
        images_dir = batch_dir / "images"

        # Collect images
        items_to_infer = []
        for item in catalog:
            img_path = images_dir / item["image_filename"]
            items_to_infer.append((item, img_path))

        # Batch inference
        predictions = []
        for i in range(0, len(items_to_infer), batch_size):
            chunk = items_to_infer[i : i + batch_size]
            pil_images = [Image.open(p).convert("RGB") for _, p in chunk]
            pixel_values = processor(images=pil_images, return_tensors="pt").pixel_values.to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values,
                    max_length=128,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.12,
                )

            decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
            for text in decoded:
                predictions.append(clean_ocr_text(text))

        # Update catalog with predictions
        for idx_item, (item, _) in enumerate(items_to_infer):
            item["ground_truth"] = predictions[idx_item]

        # Save updated catalog JSON
        cat_file = batch_dir / f"{stem}_lines.json"
        with open(cat_file, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2)

        # Save updated dataset.csv
        csv_file = batch_dir / "dataset.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_filename", "ground_truth", "page", "line", "stroke_count"])
            for item in catalog:
                writer.writerow([
                    item["image_filename"],
                    item["ground_truth"],
                    item["page"],
                    item["line"],
                    item["stroke_count"],
                ])

        elapsed = time.time() - b_t0
        print(f"    -> Done in {elapsed:.1f}s. Sample line: {repr(predictions[0] if predictions else '')}")

    total_time = time.time() - t_start
    mins = int(total_time // 60)
    secs = int(total_time % 60)
    print("\n" + "=" * 66)
    print(f"✨ Wave 3 Staging Complete: {len(staged_batches)} batches staged!")
    print(f"📊 Total lines pre-transcribed: {total_lines}")
    print(f"⏱️  Total GPU time: {mins}m {secs}s")
    print("==================================================================\n")


if __name__ == "__main__":
    main()
