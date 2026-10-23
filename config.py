"""
config.py - Central Configuration & Environment Loader for Handwriting Recognition.

Reads settings from .env file (if present) or environment variables,
with sensible local fallbacks.
"""

import os
from pathlib import Path


def load_env_file(env_file: str = ".env"):
    """Load key-value pairs from .env into os.environ (if not already set)."""
    p = Path(env_file)
    if not p.exists():
        # Check parent directory as fallback
        p = Path("..") / env_file
    if not p.exists():
        return

    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


# Automatically load on module import
load_env_file()

# Standard paths
DATA_DIR = Path(os.environ.get("HANDWRITING_DATA_DIR", "../handwriting-data")).resolve()
MODEL_DIR = Path(
    os.environ.get(
        "HANDWRITING_MODEL_DIR",
        str(DATA_DIR / "models/best_model")
        if (DATA_DIR / "models/best_model").exists()
        else "trocr_finetuned/best_model",
    )
).resolve()
VAULT_DIR = Path(os.environ.get("OBSIDIAN_VAULT_DIR", Path.home() / "Documents/obsidian/main")).resolve()
BACKUP_TARBALL = Path(
    os.environ.get(
        "HANDWRITING_BACKUP_TARBALL",
        str(VAULT_DIR / "notepads/transcriptions_backup_raw_20261022.tar.gz"),
    )
).resolve()
PORT = int(os.environ.get("PORT", "8080"))
