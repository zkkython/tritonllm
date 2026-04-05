#!/usr/bin/env python3
"""Use browser automation to fetch a Medium article and save complete Markdown.

Features:
- Uses Playwright (real Chromium) instead of direct crawler requests.
- Preserves title/content/images/code/formulas as much as possible.
- Downloads remote images to local assets folder and rewrites image links.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag
from markdownify import markdownify as html_to_md
from playwright.sync_api import Page, sync_playwright


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", text)
    return re.sub(r"-+", "-", text).strip("-") or "article"


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name.strip("._") or "image"


def extract_extension(url: str, content_type: str | None) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix
    if suffix and len(suffix) <= 6:
        return suffix
    if content_type:
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if ext:
            return ext
    return ".bin"


def parse_cookie_kv(items: List[str]) -> Dict[str, str]:
    cookies: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            continue
        name, value = item.split("=", 1)
        name = name.strip()
        value = value.strip()
        if name:
            cookies[name] = value
    return cookies


def load_cookie_file(path: Path) -> List[Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_cookies = data.get("cookies") if isinstance(data, dict) else data
    if not isinstance(raw_cookies, list):
        raise ValueError("Cookie file must be a list or include a top-level 'cookies' list.")

    normalized: List[Dict[str, object]] = []
    for c in raw_cookies:
        if not isinstance(c, dict) or "name" not in c or "value" not in c:
            continue
        domain = c.get("domain") or ".medium.com"
        normalized.append(
            {
                "name": c["name"],
                "value": c["value"],
                "domain": domain,
                "path": c.get("path", "/"),
                "secure": bool(c.get("secure", True)),
                "httpOnly": bool(c.get("httpOnly", False)),
                "sameSite": c.get("sameSite", "Lax"),
            }
        )
    return normalized


def dismiss_medium_popups(page: Page) -> None:
    selectors = [
        "button[aria-label='Dismiss']",
        "button[aria-label='Close']",
        "button[data-action='dialog_dismiss']",
        "button[data-testid='headerSignUpButton']",
        "div[role='dialog'] button[aria-label='Close']",
        "div[role='dialog'] button",
        "button:has-text('Not now')",
        "button:has-text('Maybe later')",
        "button:has-text('Continue without')",
        "button:has-text('Dismiss')",
    ]
    for _ in range(3):
        for sel in selectors:
            try:
                loc = page.locator(sel)
                if loc.count() > 0 and loc.first.is_visible():
                    loc.first.click(timeout=1200)
                    page.wait_for_timeout(300)
            except Exception:
                pass
        try:
            page.keyboard.press("Escape")
        except Exception:
            pass

    # Remove persistent overlays if close buttons fail.
    try:
        page.evaluate(
            """
            () => {
              const blockers = Array.from(document.querySelectorAll(
                '[aria-modal="true"], [role="dialog"], .overlay, .metabarModal'
              ));
              blockers.forEach((el) => {
                try {
                  el.style.display = 'none';
                  el.remove();
                } catch (_) {}
              });
              document.body.style.overflow = 'auto';
              document.documentElement.style.overflow = 'auto';
            }
            """
        )
    except Exception:
        pass


def looks_like_bot_challenge(page: Page) -> bool:
    checks = [
        "Performing security verification",
        "Verification successful. Waiting for",
        "This website uses a security service",
        "Just a moment...",
        "Ray ID",
        "Cloudflare",
    ]
    try:
        title = (page.title() or "").strip()
    except Exception:
        title = ""
    try:
        body = page.inner_text("body")[:8000]
    except Exception:
        body = ""
    joined = f"{title}\n{body}"
    return any(marker in joined for marker in checks)


def wait_for_manual_verification(page: Page, timeout_ms: int) -> bool:
    start = time.time()
    while (time.time() - start) * 1000 < timeout_ms:
        if not looks_like_bot_challenge(page):
            return True
        page.wait_for_timeout(1000)
    return not looks_like_bot_challenge(page)


def fetch_with_retries(url: str, retries: int, timeout_s: int = 30) -> requests.Response | None:
    for i in range(retries + 1):
        try:
            resp = requests.get(url, timeout=timeout_s)
            resp.raise_for_status()
            return resp
        except Exception:
            if i == retries:
                return None
            time.sleep(min(6, 1.2 * (2**i)))
    return None


def download_images(soup: BeautifulSoup, page_url: str, assets_dir: Path, retries: int) -> int:
    assets_dir.mkdir(parents=True, exist_ok=True)
    seen: Dict[str, str] = {}
    count = 0

    for img in soup.find_all("img"):
        raw_src = img.get("src") or img.get("data-src")
        if not raw_src:
            srcset = img.get("srcset")
            if srcset:
                raw_src = srcset.split(",")[0].strip().split(" ")[0]
        if not raw_src:
            continue

        src = urljoin(page_url, raw_src)
        if src in seen:
            img["src"] = seen[src]
            continue

        resp = fetch_with_retries(src, retries=retries, timeout_s=30)
        if resp is None:
            continue

        ext = extract_extension(src, resp.headers.get("content-type"))
        digest = hashlib.sha1(src.encode("utf-8")).hexdigest()[:12]
        filename = sanitize_filename(f"img_{count+1}_{digest}{ext}")
        target = assets_dir / filename
        target.write_bytes(resp.content)

        local_rel = f"{assets_dir.name}/{filename}"
        img["src"] = local_rel
        seen[src] = local_rel
        count += 1

    return count


def extract_code_text(pre: Tag) -> str:
    code_node = pre.find("code") or pre

    # 1) Prefer explicit line wrappers used by many syntax highlighters.
    line_nodes = code_node.select(
        ".line, .code-line, .view-line, [data-line], [data-code-line], [class*='line-']"
    )
    if len(line_nodes) >= 2:
        lines = [ln.get_text("", strip=False).rstrip("\n\r") for ln in line_nodes]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text).strip("\n")
        if text:
            return text

    # 2) Preserve only meaningful hard breaks; avoid splitting per token/span.
    parts: List[str] = []
    block_newline_tags = {"div", "p", "li", "tr"}
    for node in code_node.descendants:
        if isinstance(node, NavigableString):
            parts.append(str(node))
            continue
        if isinstance(node, Tag):
            if node.name == "br":
                parts.append("\n")
            elif node.name in block_newline_tags:
                parts.append("\n")

    text = "".join(parts).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip("\n")
    if text:
        return text

    # 3) Fallback.
    return code_node.get_text("", strip=False).replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")


def preserve_math_and_code(soup: BeautifulSoup, code_mode: str = "fenced") -> Tuple[BeautifulSoup, List[Tuple[str, str]]]:
    placeholders: List[Tuple[str, str]] = []

    def add_placeholder(md_snippet: str) -> str:
        token = f"@@BLOCK_{len(placeholders)}@@"
        placeholders.append((token, md_snippet))
        return token

    for script in soup.select("script[type*='math/tex']"):
        expr = script.get_text().strip()
        if not expr:
            script.decompose()
            continue
        is_display = "mode=display" in (script.get("type") or "")
        md = f"$$\n{expr}\n$$" if is_display else f"${expr}$"
        script.replace_with(add_placeholder(md))

    for katex in soup.select("span.katex"):
        annotation = katex.select_one("annotation")
        if not annotation:
            continue
        expr = annotation.get_text().strip()
        if not expr:
            continue
        display_parent = katex.find_parent(class_=re.compile(r"display", re.I))
        md = f"$$\n{expr}\n$$" if display_parent else f"${expr}$"
        katex.replace_with(add_placeholder(md))

    for pre in soup.find_all("pre"):
        code_text = extract_code_text(pre)
        if not code_text:
            pre.decompose()
            continue

        lang = ""
        classes = []
        if pre.get("class"):
            classes.extend(pre.get("class"))
        code_tag = pre.find("code")
        if code_tag and code_tag.get("class"):
            classes.extend(code_tag.get("class"))

        for cls in classes:
            m = re.search(r"(?:lang|language)-([a-zA-Z0-9_+-]+)", cls)
            if m:
                lang = m.group(1)
                break

        if code_mode == "html":
            esc = (
                code_text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            class_attr = f' class="language-{lang}"' if lang else ""
            md = f"<pre><code{class_attr}>{esc}</code></pre>"
        else:
            md = f"```{lang}\n{code_text}\n```"
        pre.replace_with(add_placeholder(md))

    return soup, placeholders


def to_markdown(html: str, code_mode: str = "fenced") -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["style", "noscript", "iframe"]):
        tag.decompose()

    soup, placeholders = preserve_math_and_code(soup, code_mode=code_mode)

    md = html_to_md(
        str(soup),
        heading_style="ATX",
        bullets="-",
        escape_asterisks=False,
        escape_underscores=False,
    )

    for token, snippet in placeholders:
        md = md.replace(token, snippet)

    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return md


def build_readable_html(title: str, source_url: str, body_html: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --card: #ffffff;
      --text: #1b1f24;
      --muted: #5f6b7a;
      --line: #e8ecf2;
      --code-bg: #0f172a;
      --code-text: #e5e7eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.75;
    }}
    .wrap {{ max-width: 920px; margin: 40px auto; padding: 0 20px; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 32px;
      box-shadow: 0 8px 20px rgba(16,24,40,.06);
    }}
    h1,h2,h3,h4 {{ line-height: 1.3; margin-top: 1.5em; }}
    h1 {{ margin-top: 0; font-size: 2rem; }}
    img {{ max-width: 100%; height: auto; border-radius: 8px; display: block; margin: 1.1em auto; }}
    pre {{
      background: var(--code-bg);
      color: var(--code-text);
      overflow-x: auto;
      padding: 14px 16px;
      border-radius: 10px;
      line-height: 1.5;
      font-size: 13px;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: .95em;
    }}
    pre code {{ white-space: pre; }}
    :not(pre) > code {{
      background: #eef2ff;
      color: #1f3a8a;
      padding: .15em .4em;
      border-radius: 6px;
    }}
    blockquote {{
      border-left: 4px solid #d6deeb;
      margin: 1em 0;
      padding: .4em 1em;
      color: var(--muted);
      background: #fbfcff;
    }}
    .meta {{ color: var(--muted); font-size: 14px; margin-bottom: 18px; }}
    a {{ color: #155eef; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <main class="wrap">
    <article class="card">
      <h1>{title}</h1>
      <div class="meta">来源: <a href="{source_url}" target="_blank" rel="noopener noreferrer">{source_url}</a></div>
      {body_html}
    </article>
  </main>
</body>
</html>
"""


def try_wait_network_idle(page: Page, timeout_ms: int) -> None:
    # Some sites (including Medium) keep background requests alive for a long time.
    # Treat networkidle as a best-effort signal and continue on timeout.
    try:
        page.wait_for_load_state("networkidle", timeout=timeout_ms)
    except Exception:
        pass


def load_page_with_retries(
    page: Page,
    url: str,
    navigation_timeout_ms: int,
    idle_timeout_ms: int,
    manual_verify_timeout_ms: int,
    retries: int,
) -> None:
    for i in range(retries + 1):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=navigation_timeout_ms)
            page.wait_for_timeout(1200)

            if looks_like_bot_challenge(page):
                if manual_verify_timeout_ms > 0:
                    ok = wait_for_manual_verification(page, timeout_ms=manual_verify_timeout_ms)
                    if not ok:
                        raise RuntimeError(
                            "Detected Cloudflare/security verification page. "
                            "Manual verification timed out."
                        )
                else:
                    raise RuntimeError(
                        "Detected Cloudflare/security verification page. "
                        "Rerun with --show-browser --manual-verify-timeout-ms 120000 and complete verification."
                    )

            dismiss_medium_popups(page)
            try_wait_network_idle(page, timeout_ms=idle_timeout_ms)

            # Trigger lazy-loaded content.
            page.evaluate(
                """
                async () => {
                  for (let i = 0; i < 8; i++) {
                    window.scrollBy(0, window.innerHeight * 0.9);
                    await new Promise((r) => setTimeout(r, 280));
                  }
                  window.scrollTo(0, 0);
                }
                """
            )
            try_wait_network_idle(page, timeout_ms=min(idle_timeout_ms, 3500))
            dismiss_medium_popups(page)
            return
        except Exception:
            if i == retries:
                raise
            time.sleep(min(8, 1.5 * (2**i)))


def pick_page_from_context(context, target_url: str, use_existing_page: bool) -> Page:
    pages = context.pages
    if use_existing_page and pages:
        host = urlparse(target_url).hostname or ""
        for p in reversed(pages):
            try:
                purl = p.url or ""
            except Exception:
                purl = ""
            if host and host in purl:
                return p
        return pages[-1]
    return context.new_page()


def fetch_article_html(
    url: str,
    navigation_timeout_ms: int,
    idle_timeout_ms: int,
    manual_verify_timeout_ms: int,
    headless: bool,
    retries: int,
    cookie_file: Path | None,
    cookie_kv: Dict[str, str],
    storage_state_in: Path | None,
    storage_state_out: Path | None,
    cdp_url: str | None,
    use_existing_page: bool,
) -> Tuple[str, str, str]:
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(cdp_url) if cdp_url else p.chromium.launch(headless=headless)
        if cdp_url:
            if not browser.contexts:
                raise RuntimeError("CDP connected, but no browser context found.")
            context = browser.contexts[0]
        else:
            context_kwargs = {"viewport": {"width": 1440, "height": 2200}}
            if storage_state_in and storage_state_in.exists():
                context_kwargs["storage_state"] = str(storage_state_in)
            context = browser.new_context(**context_kwargs)

        all_cookies: List[Dict[str, object]] = []
        if cookie_file:
            all_cookies.extend(load_cookie_file(cookie_file))
        if cookie_kv:
            target = urlparse(url)
            host = target.hostname or "medium.com"
            domain = host if host.startswith(".") else f".{host}"
            all_cookies.extend(
                {
                    "name": k,
                    "value": v,
                    "domain": domain,
                    "path": "/",
                    "secure": True,
                    "httpOnly": False,
                    "sameSite": "Lax",
                }
                for k, v in cookie_kv.items()
            )
        if all_cookies:
            context.add_cookies(all_cookies)

        page = pick_page_from_context(context, target_url=url, use_existing_page=use_existing_page)
        if not use_existing_page:
            load_page_with_retries(
                page,
                url=url,
                navigation_timeout_ms=navigation_timeout_ms,
                idle_timeout_ms=idle_timeout_ms,
                manual_verify_timeout_ms=manual_verify_timeout_ms,
                retries=retries,
            )
        elif looks_like_bot_challenge(page):
            raise RuntimeError(
                "Current tab is still on Cloudflare verification page. "
                "Please complete verification in that Chrome window, ensure article is visible, then rerun."
            )

        title = ""
        for selector in [
            "meta[property='og:title']",
            "meta[name='title']",
            "article h1",
            "h1",
            "title",
        ]:
            try:
                el = page.query_selector(selector)
                if not el:
                    continue
                if selector.startswith("meta"):
                    value = el.get_attribute("content")
                else:
                    value = el.inner_text()
                if value and value.strip():
                    title = value.strip()
                    break
            except Exception:
                continue

        article_html = ""
        for selector in ["article", "main article", "main", "body"]:
            loc = page.locator(selector)
            if loc.count() > 0:
                try:
                    article_html = loc.first.inner_html()
                    if article_html.strip():
                        break
                except Exception:
                    pass

        full_html = page.content()
        if storage_state_out and not cdp_url:
            context.storage_state(path=str(storage_state_out))
        if not cdp_url:
            context.close()
        browser.close()

    return title or "Untitled", article_html or full_html, full_html


def save_markdown(
    url: str,
    out_dir: Path,
    filename: str | None,
    navigation_timeout_ms: int,
    idle_timeout_ms: int,
    manual_verify_timeout_ms: int,
    headless: bool,
    retries: int,
    cookie_file: Path | None,
    cookie_kv: Dict[str, str],
    storage_state_in: Path | None,
    storage_state_out: Path | None,
    cdp_url: str | None,
    use_existing_page: bool,
    export_html: bool,
    code_mode: str,
) -> Path:
    title, article_html, _ = fetch_article_html(
        url,
        navigation_timeout_ms=navigation_timeout_ms,
        idle_timeout_ms=idle_timeout_ms,
        manual_verify_timeout_ms=manual_verify_timeout_ms,
        headless=headless,
        retries=retries,
        cookie_file=cookie_file,
        cookie_kv=cookie_kv,
        storage_state_in=storage_state_in,
        storage_state_out=storage_state_out,
        cdp_url=cdp_url,
        use_existing_page=use_existing_page,
    )

    safe_slug = slugify(filename if filename else title)
    md_path = out_dir / f"{safe_slug}.md"
    assets_dir = out_dir / f"{safe_slug}_assets"

    soup = BeautifulSoup(article_html, "html.parser")
    download_images(soup, url, assets_dir, retries=retries)
    html_content = str(soup)

    body_md = to_markdown(html_content, code_mode=code_mode)
    final_md = f"# {title}\n\n来源: {url}\n\n{body_md}\n"

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path.write_text(final_md, encoding="utf-8")

    if export_html:
        html_path = out_dir / f"{safe_slug}.html"
        html_path.write_text(build_readable_html(title=title, source_url=url, body_html=html_content), encoding="utf-8")
    return md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a Medium page via browser automation and save Markdown locally."
    )
    parser.add_argument("url", help="Article URL")
    parser.add_argument("--out-dir", default="exports", help="Output directory (default: exports)")
    parser.add_argument("--filename", default=None, help="Optional output filename (without .md)")
    parser.add_argument("--timeout-ms", type=int, default=90000, help="Navigation timeout in milliseconds")
    parser.add_argument(
        "--idle-timeout-ms",
        type=int,
        default=6000,
        help="Best-effort network-idle wait timeout in milliseconds",
    )
    parser.add_argument("--retries", type=int, default=3, help="Retry count for page and image fetch")
    parser.add_argument(
        "--manual-verify-timeout-ms",
        type=int,
        default=0,
        help="When challenge page appears, wait this long for manual verification in visible browser.",
    )
    parser.add_argument(
        "--cookie-file",
        default=None,
        help="Path to cookies JSON file (list or {\"cookies\": [...]} format).",
    )
    parser.add_argument(
        "--cookie",
        action="append",
        default=[],
        help="Cookie pair, can be used multiple times: --cookie uid=xxx",
    )
    parser.add_argument(
        "--storage-state-in",
        default=None,
        help="Playwright storage-state JSON to preload session.",
    )
    parser.add_argument(
        "--storage-state-out",
        default=None,
        help="Where to save updated Playwright storage-state JSON after run.",
    )
    parser.add_argument(
        "--cdp-url",
        default=None,
        help="Connect to an existing Chrome/Chromium via CDP (example: http://127.0.0.1:9222).",
    )
    parser.add_argument(
        "--use-existing-page",
        action="store_true",
        help="When using --cdp-url, read current existing tab instead of navigating again.",
    )
    parser.add_argument(
        "--export-html",
        action="store_true",
        help="Also export a readable high-fidelity HTML file alongside markdown.",
    )
    parser.add_argument(
        "--code-mode",
        choices=["fenced", "html"],
        default="fenced",
        help="Code block rendering mode in markdown: fenced (default) or html.",
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Show browser window (disable headless mode)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    cookie_file = Path(args.cookie_file) if args.cookie_file else None
    cookie_kv = parse_cookie_kv(args.cookie)
    storage_state_in = Path(args.storage_state_in) if args.storage_state_in else None
    storage_state_out = Path(args.storage_state_out) if args.storage_state_out else None
    md_path = save_markdown(
        url=args.url,
        out_dir=out_dir,
        filename=args.filename,
        navigation_timeout_ms=args.timeout_ms,
        idle_timeout_ms=max(0, args.idle_timeout_ms),
        manual_verify_timeout_ms=max(0, args.manual_verify_timeout_ms),
        headless=not args.show_browser,
        retries=max(0, args.retries),
        cookie_file=cookie_file,
        cookie_kv=cookie_kv,
        storage_state_in=storage_state_in,
        storage_state_out=storage_state_out,
        cdp_url=args.cdp_url,
        use_existing_page=bool(args.use_existing_page),
        export_html=bool(args.export_html),
        code_mode=args.code_mode,
    )
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
