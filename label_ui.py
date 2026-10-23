"""
Web-based labeling interface for handwriting recognition datasets.
Zero external dependencies (Python standard library only: http.server).

Runs locally, renders line crop images and pre-transcription suggestions,
supports switching between multiple batches via a dropdown selector,
and saves ground-truth text into dataset.csv and <pdf>_lines.json.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import argparse
import csv
import json
import os
from pathlib import Path
import urllib.parse

try:
    from config import DATA_DIR, PORT
except ImportError:
    DATA_DIR = Path(".").resolve()
    PORT = 8080

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Handwriting Line Labeler</title>
<style>
  :root {
    --bg: #09090b;
    --card: #18181b;
    --border: #27272a;
    --text: #f4f4f5;
    --muted: #a1a1aa;
    --accent: #4f46e5;
    --accent-hover: #4338ca;
    --success: #10b981;
    --warning: #f59e0b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 100vh;
  }
  .container {
    width: 100%;
    max-width: 960px;
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1rem;
    flex-wrap: wrap;
    gap: 0.75rem;
  }
  h1 { font-size: 1.25rem; font-weight: 600; white-space: nowrap; }
  select {
    background: #09090b;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.45rem 0.8rem;
    font-size: 0.85rem;
    outline: none;
    max-width: 360px;
    cursor: pointer;
    transition: border-color 0.15s;
  }
  select:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.2);
  }
  .stats { font-size: 0.9rem; color: var(--muted); }
  .progress-bar-bg {
    width: 100%;
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
  }
  .progress-bar-fill {
    height: 100%;
    background: var(--accent);
    width: 0%;
    transition: width 0.2s ease;
  }
  .image-card {
    background: #ffffff;
    border-radius: 8px;
    padding: 1.5rem;
    display: flex;
    justify-content: center;
    align-items: center;
    overflow-x: auto;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    min-height: 140px;
  }
  .image-card img {
    max-width: 100%;
    height: auto;
    object-fit: contain;
    image-rendering: -webkit-optimize-contrast;
  }
  .input-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }
  .meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    color: var(--muted);
  }
  input[type="text"] {
    width: 100%;
    background: #09090b;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    color: var(--text);
    font-size: 1.15rem;
    font-family: inherit;
    outline: none;
    transition: border-color 0.15s, box-shadow 0.15s;
  }
  input[type="text"]:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.25);
  }
  .buttons {
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
  }
  button {
    background: #27272a;
    color: var(--text);
    border: 1px solid var(--border);
    padding: 0.6rem 1.2rem;
    border-radius: 6px;
    font-size: 0.95rem;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.15s;
  }
  button:hover { background: #3f3f46; }
  button.primary {
    background: var(--accent);
    border-color: var(--accent);
    color: #ffffff;
  }
  button.primary:hover { background: var(--accent-hover); }
  .shortcuts {
    font-size: 0.8rem;
    color: var(--muted);
    text-align: center;
    margin-top: 0.5rem;
  }
  kbd {
    background: #27272a;
    padding: 2px 6px;
    border-radius: 4px;
    border: 1px solid var(--border);
    font-family: monospace;
  }
</style>
</head>
<body>
<div class="container">
  <header>
    <div style="display:flex; align-items:center; gap:0.75rem; flex-wrap:wrap;">
      <h1>Handwriting Labeler</h1>
      <select id="batch-select" onchange="switchBatch(this.value)" title="Select Batch"></select>
      <select id="jump-select" onchange="jumpTo(this.value)" title="Jump directly to line"></select>
    </div>
    <div class="stats" id="stats">Loading...</div>
  </header>

  <div class="progress-bar-bg">
    <div class="progress-bar-fill" id="progress"></div>
  </div>

  <div class="image-card">
    <img id="line-img" src="" alt="Handwritten line crop">
  </div>

  <div class="input-card">
    <div class="meta">
      <span id="line-info">Line --</span>
      <span id="stroke-info">-- strokes</span>
    </div>

    <input type="text" id="text-input" placeholder="Type text in standard sentence casing..." autofocus autocomplete="off" autocorrect="off" autocapitalize="off" spellcheck="false">

    <div class="buttons">
      <button id="btn-prev" onclick="nav(-1)">◄ Previous (Cmd+Left)</button>
      <div style="display:flex; gap:0.5rem;">
        <button id="btn-skip" onclick="nav(1)">Skip (Tab)</button>
        <button class="primary" id="btn-save" onclick="saveAndNext()">Save & Next (Enter)</button>
      </div>
    </div>
  </div>

  <div class="shortcuts">
    Shortcuts: <kbd>Enter</kbd> Save & Next &nbsp;|&nbsp; <kbd>Cmd/Alt + ←</kbd> Prev &nbsp;|&nbsp; <kbd>Cmd/Alt + →</kbd> Next
  </div>
</div>

<script>
let items = [];
let currentIndex = 0;

async function loadBatches() {
  try {
    const res = await fetch('/api/batches');
    const data = await res.json();
    const sel = document.getElementById('batch-select');
    sel.innerHTML = '';
    data.batches.forEach(b => {
      const opt = document.createElement('option');
      opt.value = b.name;
      const marker = b.is_complete ? '✓' : '○';
      opt.textContent = `${marker} ${b.name} (${b.labeled}/${b.total})`;
      if (b.name === data.current) opt.selected = true;
      sel.appendChild(opt);
    });
  } catch(e) {
    console.error('Error loading batches:', e);
  }
}

async function switchBatch(batchName) {
  try {
    await fetch('/api/switch_batch', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({batch: batchName})
    });
    await loadCatalog();
    await loadBatches();
  } catch(e) {
    console.error('Error switching batch:', e);
  }
}

function populateDropdown() {
  const sel = document.getElementById('jump-select');
  sel.innerHTML = '';
  items.forEach((it, idx) => {
    const opt = document.createElement('option');
    opt.value = idx;
    const hasText = it.ground_truth && it.ground_truth.trim().length > 0;
    const marker = hasText ? '✓' : '○';
    const preview = hasText ? ` : ${it.ground_truth.slice(0, 30)}` : ' (empty)';
    opt.textContent = `${marker} P${it.page} L${String(it.line).padStart(2, '0')}${preview}`;
    sel.appendChild(opt);
  });
}

function updateDropdownOption(idx) {
  const sel = document.getElementById('jump-select');
  if (sel && sel.options[idx]) {
    const it = items[idx];
    const hasText = it.ground_truth && it.ground_truth.trim().length > 0;
    const marker = hasText ? '✓' : '○';
    const preview = hasText ? ` : ${it.ground_truth.slice(0, 30)}` : ' (empty)';
    sel.options[idx].textContent = `${marker} P${it.page} L${String(it.line).padStart(2, '0')}${preview}`;
  }
}

function jumpTo(idx) {
  const n = parseInt(idx, 10);
  if (!isNaN(n) && n >= 0 && n < items.length) {
    currentIndex = n;
    renderCurrent();
  }
}

async function loadCatalog() {
  const res = await fetch('/api/catalog');
  items = await res.json();
  populateDropdown();
  // Find first unlabeled item
  const firstUnlabeled = items.findIndex(it => !it.ground_truth || it.ground_truth.trim() === '');
  currentIndex = firstUnlabeled >= 0 ? firstUnlabeled : 0;
  renderCurrent();
}

function renderCurrent() {
  if (items.length === 0) {
    document.getElementById('stats').textContent = "No lines found in batch";
    return;
  }

  const it = items[currentIndex];
  document.getElementById('line-img').src = `/api/image?path=${encodeURIComponent(it.image_filename)}`;
  document.getElementById('text-input').value = it.ground_truth || "";
  document.getElementById('line-info').textContent = `Page ${it.page}, Line ${it.line} (${it.image_filename})`;
  document.getElementById('stroke-info').textContent = `${it.stroke_count} strokes`;

  const labeledCount = items.filter(x => x.ground_truth && x.ground_truth.trim().length > 0).length;
  const pct = Math.round((labeledCount / items.length) * 100);
  document.getElementById('stats').textContent = `${currentIndex + 1} / ${items.length} (${labeledCount} labeled, ${pct}%)`;
  document.getElementById('progress').style.width = `${pct}%`;

  const sel = document.getElementById('jump-select');
  if (sel) sel.value = currentIndex;

  const inp = document.getElementById('text-input');
  inp.focus();
  inp.select();
}

async function saveAndNext() {
  if (items.length === 0) return;
  const text = document.getElementById('text-input').value.trim();
  items[currentIndex].ground_truth = text;
  updateDropdownOption(currentIndex);

  fetch('/api/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ index: currentIndex, ground_truth: text })
  });

  if (currentIndex + 1 < items.length) {
    currentIndex++;
    renderCurrent();
  } else {
    renderCurrent();
    loadBatches();
    alert('Finished all lines in this batch! You can switch to the next batch from the dropdown.');
  }
}

function nav(delta) {
  const next = currentIndex + delta;
  if (next >= 0 && next < items.length) {
    currentIndex = next;
    renderCurrent();
  }
}

document.getElementById('text-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    saveAndNext();
  } else if ((e.metaKey || e.altKey) && e.key === 'ArrowLeft') {
    e.preventDefault();
    nav(-1);
  } else if ((e.metaKey || e.altKey) && e.key === 'ArrowRight') {
    e.preventDefault();
    nav(1);
  }
});

loadBatches();
loadCatalog();
</script>
</body>
</html>
"""


class LabelHandler(BaseHTTPRequestHandler):
    catalog = []
    catalog_path = None
    csv_path = None
    current_batch_name = ""
    base_dir = Path(".")

    def log_message(self, format, *args):
        pass  # Quiet logs for clean terminal

    @classmethod
    def load_batch(cls, batch_dir: Path):
        cls.current_batch_name = batch_dir.name
        # Find json catalog
        jsons = list(batch_dir.glob("*_lines.json"))
        cls.catalog_path = str(jsons[0].resolve()) if jsons else None
        cls.csv_path = str((batch_dir / "dataset.csv").resolve())

        if cls.catalog_path and Path(cls.catalog_path).exists():
            with open(cls.catalog_path, "r", encoding="utf-8") as f:
                cls.catalog = json.load(f)
        else:
            cls.catalog = []

        if cls.csv_path and Path(cls.csv_path).exists():
            with open(cls.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                labels_map = {row["image_filename"]: row.get("ground_truth", "") for row in reader}
                for item in cls.catalog:
                    if item["image_filename"] in labels_map and labels_map[item["image_filename"]]:
                        item["ground_truth"] = labels_map[item["image_filename"]]

    @classmethod
    def discover_batches(cls):
        batches = []
        candidates = []
        if (cls.base_dir / "batches").is_dir():
            candidates.extend(sorted((cls.base_dir / "batches").glob("batch_*")))
        candidates.extend(sorted(cls.base_dir.glob("batch_*")))

        seen = set()
        for d in candidates:
            if d.is_dir() and d.name not in seen:
                seen.add(d.name)
                csv_path = d / "dataset.csv"
                total = 0
                labeled = 0
                if csv_path.exists():
                    with open(csv_path, "r", encoding="utf-8") as f:
                        rows = list(csv.DictReader(f))
                        total = len(rows)
                        labeled = sum(1 for r in rows if r.get("ground_truth", "").strip())
                batches.append({
                    "name": d.name,
                    "path": str(d.resolve()),
                    "total": total,
                    "labeled": labeled,
                    "is_complete": (total > 0 and labeled == total),
                })
        return batches

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif parsed.path == "/api/batches":
            batches = self.discover_batches()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"batches": batches, "current": self.current_batch_name}).encode("utf-8"))

        elif parsed.path == "/api/catalog":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.catalog).encode("utf-8"))

        elif parsed.path == "/api/image":
            query = urllib.parse.parse_qs(parsed.query)
            raw_path = query.get("path", [""])[0]

            resolved_file = None
            candidates = [
                Path(raw_path),
                self.base_dir / raw_path,
                self.base_dir / "batches" / raw_path,
                Path(raw_path.replace("handwriting-recognition/", "")),
                Path("handwriting-recognition") / raw_path,
            ]
            if self.catalog_path:
                cat_dir = Path(self.catalog_path).resolve().parent
                candidates.extend([
                    cat_dir / raw_path,
                    cat_dir / Path(raw_path).name,
                    cat_dir / "images" / Path(raw_path).name,
                    cat_dir.parent / raw_path,
                ])

            for cand in candidates:
                if cand.is_file():
                    resolved_file = cand
                    break

            if resolved_file:
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(resolved_file, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/switch_batch":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body.decode("utf-8"))
            b_name = data.get("batch")
            target = self.base_dir / b_name
            if not target.is_dir():
                target = self.base_dir / "batches" / b_name
            if target.is_dir():
                self.load_batch(target)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            else:
                self.send_response(404)
                self.end_headers()

        elif parsed.path == "/api/save":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body.decode("utf-8"))
            idx = data.get("index")
            text = data.get("ground_truth", "")

            if 0 <= idx < len(self.catalog):
                self.catalog[idx]["ground_truth"] = text

                # Save updated JSON
                if self.catalog_path:
                    with open(self.catalog_path, "w", encoding="utf-8") as f:
                        json.dump(self.catalog, f, indent=2)

                # Save / sync dataset.csv
                if self.csv_path:
                    self.save_csv()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

    @classmethod
    def save_csv(cls):
        if not cls.csv_path:
            return
        with open(cls.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_filename", "ground_truth", "page", "line", "stroke_count"])
            for item in cls.catalog:
                writer.writerow([
                    item["image_filename"],
                    item.get("ground_truth", ""),
                    item.get("page", 1),
                    item.get("line", 1),
                    item.get("stroke_count", 0),
                ])


def main():
    parser = argparse.ArgumentParser(description="Web labeling tool for handwriting line crops.")
    parser.add_argument("--catalog", default=None, help="Path to *_lines.json catalog file (optional)")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help=f"Data directory containing batches (default: {DATA_DIR})")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port to serve on (default: {PORT})")
    args = parser.parse_args()

    LabelHandler.base_dir = Path(args.data_dir).resolve()

    if args.catalog:
        cat_path = Path(args.catalog).resolve()
        LabelHandler.load_batch(cat_path.parent)
    else:
        batches = LabelHandler.discover_batches()
        if not batches:
            print(f"Error: No batch_* directories found in {LabelHandler.base_dir}")
            return
        # Default to first batch (or first incomplete batch)
        first_target = next((b for b in batches if not b["is_complete"]), batches[0])
        LabelHandler.load_batch(Path(first_target["path"]))

    server = HTTPServer((args.host, args.port), LabelHandler)
    display_host = "localhost" if args.host in ("0.0.0.0", "127.0.0.1") else args.host
    print(f"\n=======================================================")
    print(f"  Handwriting Labeling Tool running at:")
    print(f"  http://{display_host}:{args.port}")
    if args.host == "0.0.0.0":
        print(f"  (Listening on all interfaces: access via Tailscale/LAN IP)")
    print(f"=======================================================")
    print(f"Loaded batch: {LabelHandler.current_batch_name} ({len(LabelHandler.catalog)} lines)")
    print(f"Auto-saving to {LabelHandler.csv_path}")
    print(f"Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping labeling server.")
        LabelHandler.save_csv()


if __name__ == "__main__":
    main()
