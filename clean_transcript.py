"""
clean_transcript.py - Constrained, Visually-Grounded Post-Processing for TrOCR Outputs.

Features:
1. Personal Vault Lexicon + English Dictionary: Combines 160k+ words from Obsidian vault
   with standard English lexicon and personal proper nouns (Catsby, Cheddars, Lamy, AL-star).
2. Frontmatter Protection: Markdown frontmatter between --- and --- is completely untouched.
3. Levenshtein Hard Clamp: Hard programmatic constraint (max edit distance <= 2).
   Physically prevents hallucinatory smoothing: authentic idiosyncratic phrasing (e.g. 'awhile frettingly')
   is mathematically protected from being rewritten into generic clichés.
4. Valid-Word OCR Confusion Disambiguation (e.g. 'find' -> 'fine', 'next' -> 'text', 'Good' -> 'Could').
"""

import argparse
from collections import Counter
import difflib
import glob
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

try:
    from config import DATA_DIR, VAULT_DIR
except ImportError:
    DATA_DIR = Path("../handwriting-data").resolve()
    VAULT_DIR = Path.home() / "Documents/obsidian/main"

# Known personal hardware, brands, pets, and collaborators
CUSTOM_PROPER_NOUNS = {
    "lamy", "al-star", "safari", "catsby", "cheddars", "mozilla", "dan", "hon",
    "e-ink", "boox", "trocr", "dialtones", "zack", "freedman", "sanjay", "mortimer",
    "palworld", "github", "claude", "portia", "decafclaw"
}

# Architectural small-caps glyph confusion pairs with low visual penalty
SMALL_CAPS_CONFUSION_PAIRS = {
    ("c", "g"): 0.5,
    ("g", "c"): 0.5,
    ("d", "e"): 0.5,
    ("e", "d"): 0.5,
    ("c", "e"): 0.5,
    ("e", "c"): 0.5,
    ("n", "t"): 0.5,
    ("t", "n"): 0.5,
    ("f", "t"): 0.5,
    ("t", "f"): 0.5,
    ("b", "s"): 0.6,
    ("s", "b"): 0.6,
    ("v", "u"): 0.5,
    ("u", "v"): 0.5,
    ("l", "i"): 0.4,
    ("i", "l"): 0.4,
    ("1", "l"): 0.3,
    ("0", "o"): 0.3,
}

# High-confidence visual confusion replacements in specific grammatical contexts
CONTEXT_REPLACEMENTS = {
    # e.g. "light is find" -> "light is fine"
    ("is", "find"): "fine",
    ("was", "find"): "fine",
    ("feeling", "find"): "fine",
    ("seems", "find"): "fine",
    # e.g. "convert to next" -> "convert to text"
    ("to", "next"): "text",
    ("into", "next"): "text",
    ("plain", "next"): "text",
    ("raw", "next"): "text",
    # e.g. "Good be more like" -> "Could be more like"
    ("start_of_sentence", "good"): "Could",
    # e.g. "think table" -> "e-ink tablet"
    ("new", "think"): "e-ink",
    ("think", "table"): "tablet",
    ("e-ink", "table"): "tablet",
    # e.g. "reflective —ink" -> "reflective e-ink"
    ("reflective", "—ink"): "e-ink",
    ("reflective", "-ink"): "e-ink",
    # e.g. "Catsby seeing" -> "Catsby being"
    ("catsby", "seeing"): "being",
    ("still", "seeing"): "being",
}


def damerau_levenshtein(s1: str, s2: str) -> int:
    """Compute Damerau-Levenshtein distance between two strings."""
    d = {}
    len1, len2 = len(s1), len(s2)
    for i in range(-1, len1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, len2 + 1):
        d[(-1, j)] = j + 1

    for i in range(len1):
        for j in range(len2):
            cost = 0 if s1[i] == s2[j] else 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,        # deletion
                d[(i, j - 1)] + 1,        # insertion
                d[(i - 1, j - 1)] + cost,  # substitution
            )
            if i > 0 and j > 0 and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[(i - 2, j - 2)] + 1)  # transposition

    return d[(len1 - 1, len2 - 1)]


class LexiconModel:
    """Manages vault vocabulary, English dictionary, and n-gram frequency distributions."""

    def __init__(self, vocab_cache_path: str = "vault_vocab.json"):
        self.vocab_cache = Path(vocab_cache_path)
        self.unigrams: Counter = Counter()
        self.bigrams: Counter = Counter()
        self.english_words: Set[str] = set()
        self.total_words = 0

    def load_or_build(self, vault_roots: Optional[List[str]] = None):
        # 1. Load English dictionary words
        dict_candidates = [
            DATA_DIR / "vocab/english_words.txt",
            DATA_DIR / "english_words.txt",
            Path("handwriting-recognition/english_words.txt"),
            Path("english_words.txt"),
            Path("/usr/share/dict/words"),
        ]
        for dc in dict_candidates:
            if dc.exists():
                try:
                    with open(dc, "r", encoding="utf-8", errors="replace") as f:
                        self.english_words = {line.strip().lower() for line in f if line.strip()}
                    print(f"Loaded {len(self.english_words)} words from English dictionary ({dc.name}).")
                    break
                except Exception:
                    pass

        # 2. Load vault vocabulary cache
        cache_candidates = [
            self.vocab_cache,
            DATA_DIR / "vocab/vault_vocab.json",
            DATA_DIR / "vault_vocab.json",
        ]
        target_cache = next((c for c in cache_candidates if c.exists()), None)
        if target_cache:
            print(f"Loading vocabulary cache from {target_cache}...")
            try:
                with open(target_cache, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.unigrams = Counter(data.get("unigrams", {}))
                    self.bigrams = Counter(data.get("bigrams", {}))
                    self.total_words = sum(self.unigrams.values())
                    print(f"  Loaded {len(self.unigrams)} unigrams, {len(self.bigrams)} bigrams.")
                    return
            except Exception as e:
                print(f"  Warning: Failed to load cache ({e}), rebuilding...")

        # Build vocabulary from vault
        if not vault_roots:
            candidates = [
                VAULT_DIR,
                Path.home() / ".decafclaw/decafclaw/obsidian/main",
                Path.home() / "Documents/obsidian/main",
                Path("obsidian/main"),
            ]
            vault_roots = [str(p) for p in candidates if p.exists()]

        print("Building personal lexicon from vault markdown files...")
        word_re = re.compile(r"\b[A-Za-z0-9'’_-]+\b")
        files_scanned = 0

        for vr in vault_roots:
            vr_path = Path(vr)
            for md_file in vr_path.rglob("*.md"):
                if "notepads/transcriptions" in str(md_file):
                    continue
                files_scanned += 1
                try:
                    with open(md_file, "r", encoding="utf-8", errors="replace") as f:
                        text = f.read()
                    words = [w.lower() for w in word_re.findall(text)]
                    self.unigrams.update(words)
                    for i in range(len(words) - 1):
                        self.bigrams[f"{words[i]} {words[i+1]}"] += 1
                except Exception:
                    pass

        # Ingest verified batch ground truths
        batch_files = glob.glob("batch_*/dataset.csv")
        for bf in batch_files:
            try:
                import csv
                with open(bf, "r", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        gt = row.get("ground_truth", "")
                        words = [w.lower() for w in word_re.findall(gt)]
                        self.unigrams.update(words)
                        for i in range(len(words) - 1):
                            self.bigrams[f"{words[i]} {words[i+1]}"] += 1
            except Exception:
                pass

        self.total_words = sum(self.unigrams.values())
        print(f"Scanned {files_scanned} notes: {len(self.unigrams)} unique words, {self.total_words} total tokens.")

        try:
            with open(self.vocab_cache, "w", encoding="utf-8") as f:
                json.dump({
                    "unigrams": dict(self.unigrams.most_common(60000)),
                    "bigrams": dict(self.bigrams.most_common(100000)),
                }, f)
            print(f"Saved vocabulary cache to {self.vocab_cache}")
        except Exception as e:
            print(f"Warning: Failed to save vocab cache: {e}")

    def is_valid_word(self, word: str) -> bool:
        """Check if token is a recognized word in English, personal lexicon, or proper nouns."""
        w = word.lower().strip(".,!?:;\"'()[]{}—-")
        if not w:
            return True
        if w in self.english_words or w in CUSTOM_PROPER_NOUNS:
            return True
        if self.unigrams.get(w, 0) >= 3:
            return True
        # Check standard inflectional suffixes
        if w.endswith("ed") and (w[:-2] in self.english_words or w[:-1] in self.english_words):
            return True
        if w.endswith("ing") and (w[:-3] in self.english_words or (w[:-3] + "e") in self.english_words):
            return True
        if w.endswith("ly") and w[:-2] in self.english_words:
            return True
        if w.endswith("s") and (w[:-1] in self.english_words or w[:-2] in self.english_words):
            return True
        # Check hyphenated compounds
        if "-" in w:
            parts = [p for p in w.split("-") if p]
            if all(p in self.english_words or p in CUSTOM_PROPER_NOUNS or self.unigrams.get(p, 0) >= 2 for p in parts):
                return True
        return False


def clean_line_text(
    line: str,
    lexicon: LexiconModel,
    max_edit_distance: int = 2,
) -> Tuple[str, List[Tuple[str, str, str]]]:
    """Clean OCR line with strictly bounded Levenshtein edits."""
    stripped = line.strip()
    if not stripped or stripped.startswith("---") or stripped.startswith("#") or stripped.startswith("model:"):
        return line, []

    changes = []
    tokens = re.findall(r"\S+|\s+", line)
    new_tokens = list(tokens)

    for i, tok in enumerate(tokens):
        if not tok.strip():
            continue

        match = re.match(r"^([^A-Za-z0-9]*)([A-Za-z0-9'’_-]+)([^A-Za-z0-9]*)$", tok)
        if not match:
            continue

        prefix, word, suffix = match.groups()
        word_lower = word.lower()

        # Check preceding word for context
        prev_word = ""
        for j in range(i - 1, -1, -1):
            if tokens[j].strip():
                prev_match = re.match(r"^[^A-Za-z0-9]*([A-Za-z0-9'’_-]+)", tokens[j].strip())
                if prev_match:
                    prev_word = prev_match.group(1).lower()
                break

        # Check high-confidence grammatical OCR context replacements
        context_key = (prev_word, word_lower) if prev_word else ("start_of_sentence", word_lower)
        if context_key in CONTEXT_REPLACEMENTS:
            replacement = CONTEXT_REPLACEMENTS[context_key]
            if word.istitle():
                replacement = replacement.capitalize()
            elif word.isupper():
                replacement = replacement.upper()

            dist = damerau_levenshtein(word_lower, replacement.lower())
            if dist <= max_edit_distance:
                new_tokens[i] = f"{prefix}{replacement}{suffix}"
                changes.append((word, replacement, f"context rule ('{prev_word} {word}' -> '{prev_word} {replacement}', dist={dist})"))
                continue

        # If word is a recognized valid English or vault word, PROTECT it
        if lexicon.is_valid_word(word_lower):
            continue

        # If word is unrecognized: look for valid English / vault candidate within distance <= max_edit_distance
        candidates = []
        word_len = len(word_lower)
        allowed_dist = 1 if word_len <= 4 else max_edit_distance

        # Candidate pool: union of known English and personal vocabulary
        pool = list(lexicon.english_words | set(lexicon.unigrams.keys()) | CUSTOM_PROPER_NOUNS)
        possible_matches = difflib.get_close_matches(word_lower, pool, n=6, cutoff=0.65)

        for cand in possible_matches:
            dist = damerau_levenshtein(word_lower, cand)
            if dist <= allowed_dist:
                freq = lexicon.unigrams.get(cand, 1)
                # Boost if cand is a known common English word or in vault
                score = math.log(freq + 1) - (dist * 1.5)
                candidates.append((cand, dist, score, freq))

        if candidates:
            candidates.sort(key=lambda x: x[2], reverse=True)
            best_cand, best_dist, best_score, best_freq = candidates[0]

            rep_text = best_cand
            if word.istitle():
                rep_text = rep_text.capitalize()
            elif word.isupper():
                rep_text = rep_text.upper()

            new_tokens[i] = f"{prefix}{rep_text}{suffix}"
            changes.append((word, rep_text, f"spelling correction (dist={best_dist}, freq={best_freq})"))

    cleaned_line = "".join(new_tokens)
    return cleaned_line, changes


def clean_markdown_document(
    input_path: Path,
    lexicon: LexiconModel,
    max_edit_distance: int = 2,
) -> Tuple[str, List[Tuple[int, str, str, str]]]:
    """Clean a full Markdown file line-by-line while strictly protecting frontmatter."""
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    cleaned_lines = []
    doc_changes = []
    in_frontmatter = False
    frontmatter_closed = False

    for line_idx, line in enumerate(lines, 1):
        stripped = line.strip()
        # Frontmatter fence detection (strictly at the beginning of the file)
        if not frontmatter_closed:
            if line_idx == 1 and stripped == "---":
                in_frontmatter = True
                cleaned_lines.append(line)
                continue
            elif in_frontmatter and stripped == "---":
                in_frontmatter = False
                frontmatter_closed = True
                cleaned_lines.append(line)
                continue
            elif in_frontmatter:
                cleaned_lines.append(line)
                continue

        cleaned, line_changes = clean_line_text(line, lexicon, max_edit_distance)
        cleaned_lines.append(cleaned)
        for orig, rep, reason in line_changes:
            doc_changes.append((line_idx, orig, rep, reason))

    return "".join(cleaned_lines), doc_changes


def main():
    parser = argparse.ArgumentParser(
        description="Constrained, Visually-Grounded Post-Processing for TrOCR Handwriting Outputs."
    )
    parser.add_argument("path", help="Path to Markdown file or directory to clean")
    parser.add_argument("--diff-only", action="store_true", help="Print diff of proposed corrections without writing")
    parser.add_argument("--write", action="store_true", help="Write cleaned file to <stem>.clean.md")
    parser.add_argument("--in-place", action="store_true", help="Overwrite file in-place (keeps backup in .raw.md)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite in-place without .raw.md (e.g. when tarball exists)")
    parser.add_argument("--max-distance", type=int, default=2, help="Max Levenshtein edit distance allowed (default: 2)")
    parser.add_argument("--vault-root", help="Path to Obsidian vault root to build lexicon")

    args = parser.parse_args()

    lexicon = LexiconModel()
    lexicon.load_or_build(vault_roots=[args.vault_root] if args.vault_root else None)

    target_path = Path(args.path)
    if target_path.is_file():
        files = [target_path]
    elif target_path.is_dir():
        files = sorted([
            f for f in target_path.rglob("*.md")
            if not (f.name.endswith(".clean.md") or f.name.endswith(".raw.md"))
        ])
    else:
        print(f"Error: Path not found: {target_path}")
        sys.exit(1)

    print(f"\nProcessing {len(files)} file(s) with Levenshtein clamp <= {args.max_distance}...\n")

    for fpath in files:
        if len(files) > 1 and (fpath.name.endswith(".clean.md") or fpath.name.endswith(".raw.md")):
            continue

        cleaned_text, changes = clean_markdown_document(fpath, lexicon, max_edit_distance=args.max_distance)

        print("=" * 65)
        print(f"File: {fpath.name} ({len(changes)} proposed correction(s))")
        print("=" * 65)

        if not changes:
            print("  No typos detected! Text conforms to vault lexicon.")
            continue

        for l_num, orig, rep, reason in changes:
            print(f"  Line {l_num:02d}: '{orig}' -> '{rep}'  [{reason}]")

        if args.write:
            out_file = fpath.with_name(f"{fpath.stem}.clean.md")
            with open(out_file, "w", encoding="utf-8") as out_f:
                out_f.write(cleaned_text)
            print(f"\n  -> Cleaned file written to: {out_file.name}")
        elif args.in_place:
            raw_backup = fpath.with_name(f"{fpath.stem}.raw.md")
            if not raw_backup.exists():
                fpath.rename(raw_backup)
            with open(fpath, "w", encoding="utf-8") as out_f:
                out_f.write(cleaned_text)
            print(f"\n  -> Updated {fpath.name} in-place (backup saved to {raw_backup.name})")
        elif args.overwrite:
            with open(fpath, "w", encoding="utf-8") as out_f:
                out_f.write(cleaned_text)
            print(f"\n  -> Overwrote {fpath.name} with cleaned text")

    print("\nDone!\n")


if __name__ == "__main__":
    main()
