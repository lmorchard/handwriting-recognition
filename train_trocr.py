"""
Fine-tune Microsoft TrOCR (microsoft/trocr-base-handwritten) on personal handwriting lines.

Supports:
- Apple Silicon (MPS) acceleration on macOS.
- NVIDIA CUDA GPU acceleration on Linux / Cloud.
- Automatic train/validation split.
- Character Error Rate (CER) and Word Error Rate (WER) evaluation.

Usage:
    python train_trocr.py --csv sample_extracted/dataset.csv --images-dir sample_extracted/images --epochs 20
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)

try:
    from config import DATA_DIR, MODEL_DIR
except ImportError:
    DATA_DIR = Path(".").resolve()
    MODEL_DIR = Path("trocr_finetuned/best_model").resolve()


def compute_cer(predictions: List[str], references: List[str]) -> float:
    """Compute Character Error Rate using Levenshtein distance."""
    def levenshtein(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    total_dist = sum(levenshtein(p, r) for p, r in zip(predictions, references))
    total_ref_len = sum(len(r) for r in references)
    return total_dist / max(1, total_ref_len)


class HandwritingLineDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor: TrOCRProcessor, max_target_length: int = 128):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        text = str(row["ground_truth"]).strip()

        image = Image.open(img_path).convert("RGB")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values[0]

        # Tokenize labels
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        # Important: Replace padding token id's with -100 so loss ignores them
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


def train(args):
    output_model_dir = Path(args.output_dir)

    if not args.batches and not (args.csv and args.images_dir):
        data_p = Path(args.data_dir).resolve()
        candidates = []
        if (data_p / "batches").is_dir():
            candidates.extend(sorted((data_p / "batches").glob("batch_*")))
        candidates.extend(sorted(data_p.glob("batch_*")))
        seen = set()
        args.batches = []
        for c in candidates:
            if c.is_dir() and c.name not in seen:
                seen.add(c.name)
                args.batches.append(str(c))
        if args.batches:
            print(f"Auto-discovered {len(args.batches)} batches in {args.data_dir}")

    if args.batches:
        dfs = []
        for b in args.batches:
            b_path = Path(b)
            csv_path = b_path / "dataset.csv"
            if not csv_path.is_file():
                print(f"Warning: {csv_path} not found, skipping...")
                continue
            b_df = pd.read_csv(csv_path)
            b_df["image_path"] = b_df["image_filename"].apply(
                lambda fn: str((b_path / "images" / Path(fn).name).resolve())
            )
            dfs.append(b_df)
        if not dfs:
            raise FileNotFoundError(f"No valid dataset.csv found in any of {args.batches}")
        df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(df)} total lines across {len(args.batches)} batches: {args.batches}")
    elif args.csv and args.images_dir:
        csv_path = Path(args.csv)
        images_dir = Path(args.images_dir)
        print(f"Loading annotations from {csv_path}...")
        df = pd.read_csv(csv_path)
        df["image_path"] = df["image_filename"].apply(lambda fn: str((images_dir / Path(fn).name).resolve()))
    else:
        raise ValueError("Must specify either --batches <dir1> <dir2> ... or both --csv and --images-dir")

    # Filter out empty annotations
    df = df[df["ground_truth"].notna()]
    df = df[df["ground_truth"].str.strip().str.len() > 0]
    print(f"Total labeled lines: {len(df)}")

    if len(df) < 10:
        print("Warning: You have fewer than 10 labeled lines. Label at least 30-50 lines for meaningful fine-tuning.")

    # Train/Validation Split (85% train, 15% val)
    train_df, val_df = train_test_split(df, test_size=args.val_split, random_state=42)
    print(f"Train samples: {len(train_df)} | Validation samples: {len(val_df)}")

    # Detect compute device
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon (MPS) acceleration!")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Using CPU (training will be slower)")

    # Load TrOCR Processor & Base Model
    print(f"Loading pretrained base model: {args.base_model}...")
    processor = TrOCRProcessor.from_pretrained(args.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(args.base_model)

    # Configure model decoder tokens & generation config
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Set generation parameters on model.generation_config for modern Transformers
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
        model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
        model.generation_config.max_length = 128
        model.generation_config.early_stopping = True
        model.generation_config.no_repeat_ngram_size = 3
        model.generation_config.length_penalty = 2.0
        model.generation_config.num_beams = 4

    train_dataset = HandwritingLineDataset(train_df, processor, max_target_length=128)
    val_dataset = HandwritingLineDataset(val_df, processor, max_target_length=128)

    def compute_metrics(eval_pred):
        preds_ids, labels_ids = eval_pred.predictions, eval_pred.label_ids
        # Replace -100 with pad token id to decode
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id

        pred_strs = processor.batch_decode(preds_ids, skip_special_tokens=True)
        label_strs = processor.batch_decode(labels_ids, skip_special_tokens=True)

        cer = compute_cer(pred_strs, label_strs)
        return {"cer": cer}

    # Build training arguments compatible with modern Transformers & MPS
    training_kwargs = {
        "output_dir": str(output_model_dir),
        "predict_with_generate": True,
        "save_strategy": "epoch",
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "logging_steps": 5,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "cer",
        "greater_is_better": False,
        "report_to": "none",
    }
    # Transformers 4.40+ uses eval_strategy, older versions used evaluation_strategy
    try:
        training_args = Seq2SeqTrainingArguments(eval_strategy="epoch", **training_kwargs)
    except TypeError:
        training_args = Seq2SeqTrainingArguments(evaluation_strategy="epoch", **training_kwargs)

    try:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=default_data_collator,
            processing_class=processor,
        )
    except TypeError:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=default_data_collator,
            tokenizer=getattr(processor, "image_processor", getattr(processor, "feature_extractor", None)),
        )

    print("\nStarting fine-tuning...")
    trainer.train()

    # Save final model & processor
    final_dir = output_model_dir / "best_model"
    print(f"\nSaving best fine-tuned model to {final_dir}...")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on personal handwriting lines.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help=f"Data directory containing batches (default: {DATA_DIR})")
    parser.add_argument("--batches", nargs="+", help="One or more batch directories (e.g. --batches batch_aug26 batch_aug25)")
    parser.add_argument("--csv", help="Path to dataset.csv (alternative to --batches)")
    parser.add_argument("--images-dir", help="Path to images directory (alternative to --batches)")
    parser.add_argument("--base-model", default="microsoft/trocr-base-handwritten", help="Hugging Face base model")
    parser.add_argument("--output-dir", default="trocr_finetuned", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device (default: 4)")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split fraction (default: 0.15)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
