# ==============================================================================
# Makefile for Handwriting Recognition Pipeline
# ==============================================================================
# Decoupled tooling for architectural small-caps TrOCR handwriting recognition.
# Data is kept separate from code via $(DATA_DIR) (default: ../handwriting-data).
# ==============================================================================

SHELL := /bin/bash
PYTHON ?= python3
UV ?= uv

# Load local environment configuration from .env if present
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

# Configuration overrides (defaults fallback to env vars or local defaults)
DATA_DIR ?= $(or $(HANDWRITING_DATA_DIR),../handwriting-data)
MODEL_DIR ?= $(or $(HANDWRITING_MODEL_DIR),$(DATA_DIR)/models/best_model)
VAULT_DIR ?= $(or $(OBSIDIAN_VAULT_DIR),$(HOME)/Documents/obsidian/main)
PORT ?= $(or $(LABEL_PORT),8080)

.PHONY: help setup setup-cpu label stage-batch train train-simmer transcribe sweep clean-transcripts harvest charts eval clean

help:
	@echo "Handwriting Recognition Pipeline Commands:"
	@echo ""
	@echo "  Environment & Setup:"
	@echo "    make setup              Install dependencies with uv (GPU / MPS default)"
	@echo "    make setup-cpu          Install CPU-only PyTorch (for lightweight Linux VMs)"
	@echo ""
	@echo "  Active Learning & Labeling:"
	@echo "    make label              Launch local zero-dependency labeling UI (port $(PORT))"
	@echo "    make stage-batch PDF=x  Extract lines and pre-transcribe a new PDF batch"
	@echo "    make harvest NOTE=x     Harvest Obsidian markdown edits into spot_corrections"
	@echo ""
	@echo "  Training:"
	@echo "    make train              Run full TrOCR fine-tuning (all batches in $(DATA_DIR))"
	@echo "    make train-simmer       Run fast 90s incremental training burst on spot_corrections"
	@echo ""
	@echo "  Transcription & Archive Ingestion:"
	@echo "    make transcribe PDF=x   Transcribe a single BOOX PDF volume to Markdown"
	@echo "    make sweep              Run unattended archive sweep across all vault PDFs"
	@echo "    make clean-transcripts  Run Levenshtein-clamped vault lexicon cleanup on transcripts"
	@echo ""
	@echo "  Visualization & Diagnostics:"
	@echo "    make charts             Regenerate CER, loss, and scaling law publication charts"
	@echo "    make eval               Evaluate model on unseen validation splits"
	@echo "    make clean              Remove temporary caches, test images, and stray logs"
	@echo ""
	@echo "Configuration:"
	@echo "  DATA_DIR  = $(DATA_DIR)"
	@echo "  MODEL_DIR = $(MODEL_DIR)"
	@echo "  VAULT_DIR = $(VAULT_DIR)"

# ------------------------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------------------------

setup:
	@echo "Installing pipeline dependencies with uv..."
	$(UV) pip install -r requirements.txt

setup-cpu:
	@echo "Installing lightweight CPU-only PyTorch for Linux VMs..."
	$(UV) pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(UV) pip install transformers pillow pypdf sentencepiece numpy

# ------------------------------------------------------------------------------
# Active Learning & Labeling
# ------------------------------------------------------------------------------

label:
	@echo "Launching Label UI on http://localhost:$(PORT) using DATA_DIR=$(DATA_DIR)..."
	$(UV) run --with-requirements requirements.txt $(PYTHON) label_ui.py --port $(PORT) --data-dir $(DATA_DIR)

stage-batch:
	@if [ -z "$(PDF)" ]; then echo "Error: Specify PDF=<path-to-journal.pdf>"; exit 1; fi
	@echo "Extracting and pre-transcribing $(PDF)..."
	$(UV) run --with-requirements requirements.txt $(PYTHON) pretranscribe.py $(PDF) --data-dir $(DATA_DIR) --model $(MODEL_DIR)

harvest:
	@if [ -z "$(NOTE)" ]; then echo "Error: Specify NOTE=<path-to-edited-note.md>"; exit 1; fi
	@echo "Harvesting edits from $(NOTE)..."
	$(UV) run --with-requirements requirements.txt $(PYTHON) harvest_corrections.py $(NOTE) --out-dir $(DATA_DIR)/spot_corrections --model $(MODEL_DIR)

# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------

train:
	@echo "Running TrOCR fine-tuning across $(DATA_DIR)..."
	$(UV) run --with-requirements requirements.txt $(PYTHON) train_trocr.py --data-dir $(DATA_DIR)

train-simmer:
	@echo "Running fast spot-correction simmer training on $(DATA_DIR)/spot_corrections..."
	$(UV) run --with-requirements requirements.txt $(PYTHON) train_simmer.py --data-dir $(DATA_DIR) --model $(MODEL_DIR)

simmer: train-simmer

# ------------------------------------------------------------------------------
# Transcription & Ingestion
# ------------------------------------------------------------------------------

transcribe:
	@if [ -z "$(PDF)" ]; then echo "Error: Specify PDF=<path-to-journal.pdf>"; exit 1; fi
	@echo "Transcribing $(PDF) with $(MODEL_DIR)..."
	$(UV) run --with-requirements requirements.txt $(PYTHON) transcribe_journal.py $(PDF) --model $(MODEL_DIR)

sweep:
	@echo "Running full unattended archive sweep on $(VAULT_DIR)/notepads/journals/*.pdf..."
	$(UV) run --with-requirements requirements.txt $(PYTHON) transcribe_journal.py \
		"$(VAULT_DIR)/notepads/journals/*.pdf" \
		-o "$(VAULT_DIR)/notepads/transcriptions" \
		--by-year \
		--skip-existing \
		--model $(MODEL_DIR)

clean-transcript:
	@if [ -z "$(FILE)" ]; then echo "Error: Specify FILE=<path-to-transcript.md>"; exit 1; fi
	$(PYTHON) clean_transcript.py "$(FILE)"

clean-transcripts:
	@echo "Running Levenshtein-clamped vault lexicon cleanup across transcripts..."
	$(PYTHON) clean_transcript.py "$(VAULT_DIR)/notepads/transcriptions" --overwrite

# ------------------------------------------------------------------------------
# Visualization & Maintenance
# ------------------------------------------------------------------------------

charts:
	@echo "Generating publication charts..."
	$(UV) run --with-requirements requirements.txt $(PYTHON) generate_charts.py

eval:
	@echo "Evaluating $(MODEL_DIR) across validation set..."
	$(UV) run --with-requirements requirements.txt $(PYTHON) eval_model.py --model $(MODEL_DIR) --data-dir $(DATA_DIR)

clean:
	@echo "Cleaning temporary files and caches..."
	rm -rf __pycache__ *.pyc .pytest_cache
	rm -f test_line_*.png page1_full.png *.raw.md
	rm -f full_sweep.log cleanup.log transcription_errors.log
