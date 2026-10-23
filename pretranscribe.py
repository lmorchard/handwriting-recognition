"""
Pre-transcribe a batch of extracted line images using the fine-tuned TrOCR model.
Enables rapid active learning: users only need to review/correct pre-filled predictions in label_ui.
"""

import argparse
import csv
import json
from pathlib import Path
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def pretranscribe_batch(catalog_path: Path, model_dir: Path, batch_size: int = 8):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {model_dir} on device: {device}...")

    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)

    catalog_path = Path(catalog_path)
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    images_dir = catalog_path.parent / "images"
    total = len(catalog)
    print(f"Pre-transcribing {total} lines with batch size {batch_size}...")

    # Process in batches
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch_items = catalog[start_idx:end_idx]

        images = []
        for item in batch_items:
            img_file = images_dir / item["image_filename"]
            images.append(Image.open(img_file).convert("RGB"))

        pixel_values = processor(images, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            outputs = model.generate(pixel_values, max_length=128)

        preds = processor.batch_decode(outputs, skip_special_tokens=True)
        for i, pred in enumerate(preds):
            batch_items[i]["ground_truth"] = pred.strip()

        print(f"  Processed {end_idx}/{total} lines...")

    # Save updated JSON catalog
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    # Save updated CSV
    csv_path = catalog_path.parent / "dataset.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_filename", "ground_truth", "page", "line", "stroke_count"])
        for item in catalog:
            writer.writerow([
                item["image_filename"],
                item.get("ground_truth", ""),
                item.get("page", 1),
                item.get("line", 1),
                item.get("stroke_count", 0),
            ])

    print(f"\nSuccessfully pre-transcribed {total} lines!")
    print(f"Catalog saved to {catalog_path}")
    print(f"CSV saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Pre-transcribe line images using fine-tuned TrOCR.")
    parser.add_argument("--catalog", required=True, help="Path to <pdf>_lines.json catalog")
    parser.add_argument("--model", default="trocr_finetuned/best_model", help="Path to fine-tuned model checkpoint")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference (default: 8)")

    args = parser.parse_args()
    pretranscribe_batch(args.catalog, args.model, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
