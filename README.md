# tritonllm

## Browser automation: Medium -> Markdown

This repository includes a script that opens pages with real Chromium via Playwright, then exports article content to local Markdown while keeping title, text, images, formulas, and code blocks as complete as possible.

### 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### 2) Export article

Set target URL first:

```bash
URL="https://example.com/your-article"
```

Then run:

```bash
python scripts/medium_to_markdown.py \
  "$URL" \
  --out-dir exports
```

If you want stable output names:

```bash
python scripts/medium_to_markdown.py \
  "$URL" \
  --out-dir exports \
  --filename article-001
```

Optional flags:

- `--filename triton-part1` custom markdown filename (without `.md`)
- `--show-browser` run with visible browser window
- `--timeout-ms 120000` navigation timeout
- `--idle-timeout-ms 4000` best-effort wait for network idle, timeout will continue
- `--retries 5` retry page load/image download with backoff
- `--cookie-file ./cookies.json` inject login cookies before loading page
- `--cookie uid=... --cookie sid=...` inject key/value cookies from CLI
- `--manual-verify-timeout-ms 120000` if security challenge appears, wait 120s for manual solve
- `--storage-state-in ./state.json` preload Playwright session state
- `--storage-state-out ./state.json` save Playwright session state after run
- `--cdp-url http://127.0.0.1:9222` connect to an existing Chrome via DevTools protocol
- `--use-existing-page` when using `--cdp-url`, scrape current tab without re-navigation
- `--export-html` also export high-fidelity readable HTML (recommended)
- `--code-mode html` preserve code blocks as raw HTML in markdown when fenced rendering is poor

### 3) Output

- Markdown file: `exports/<name>.md`
- Downloaded images: `exports/<name>_assets/*`

Image references in markdown are rewritten to local relative paths.

### Cookie file format

Use either:

1. A list:

```json
[
  { "name": "uid", "value": "...", "domain": ".medium.com", "path": "/" },
  { "name": "sid", "value": "...", "domain": ".medium.com", "path": "/" }
]
```

2. Playwright-like object:

```json
{
  "cookies": [
    { "name": "uid", "value": "...", "domain": ".medium.com", "path": "/" }
  ]
}
```

Run example with hardening options:

```bash
URL="https://example.com/your-article"

python scripts/medium_to_markdown.py \
  "$URL" \
  --out-dir exports \
  --cookie-file ./cookies.json \
  --idle-timeout-ms 4000 \
  --retries 5
```

### Cloudflare/Security challenge workflow

If you see a verification page, do this once in visible browser:

```bash
URL="https://example.com/your-article"

./.venv/bin/python scripts/medium_to_markdown.py \
  "$URL" \
  --out-dir exports \
  --show-browser \
  --manual-verify-timeout-ms 180000 \
  --storage-state-out ./medium_state.json
```

Then reuse saved session for automated runs:

```bash
URL="https://example.com/your-article"

./.venv/bin/python scripts/medium_to_markdown.py \
  "$URL" \
  --out-dir exports \
  --storage-state-in ./medium_state.json \
  --idle-timeout-ms 3000 \
  --retries 5
```

### If verification keeps expiring: attach to your real Chrome tab

1) Start Chrome with remote debugging:

```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --remote-debugging-port=9222 \
  --user-data-dir=/tmp/chrome-medium
```

2) In that Chrome window, manually open the article URL and complete verification until article content is visible.

3) Run exporter by attaching to this browser and scraping existing tab:

```bash
URL="https://example.com/your-article"

./.venv/bin/python scripts/medium_to_markdown.py \
  "$URL" \
  --out-dir exports \
  --cdp-url http://127.0.0.1:9222 \
  --use-existing-page \
  --export-html \
  --code-mode html
```

### Quick rule when changing links

- Normal case: replace `URL`, run basic command.
- Login/challenge case: use `--show-browser` + `--manual-verify-timeout-ms`, then save/reuse `--storage-state-*`.
- Challenge keeps expiring: use `--cdp-url` + `--use-existing-page` and scrape your already-open verified tab.

For best reading quality:

- Open `exports/<name>.html` for near-webpage rendering.
- Use `exports/<name>.md` for LLM/RAG indexing and text processing.
