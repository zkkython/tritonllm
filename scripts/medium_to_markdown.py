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
import sys
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


def canonical_page_url(raw: str) -> str:
    """Normalize URL for tab matching (ignore query/fragment/trailing slash)."""
    try:
        p = urlparse(raw or "")
    except Exception:
        return ""
    scheme = p.scheme or "https"
    host = (p.netloc or "").lower()
    path = (p.path or "/").rstrip("/") or "/"
    return f"{scheme}://{host}{path}"


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


def looks_like_member_preview(page: Page) -> bool:
    """Detect Medium paywalled preview state (partial content only)."""
    markers = [
        "Member-only story",
        "The rest of this story is for members only",
        "Become a member",
        "Get unlimited access",
        "Already a member?",
        "Upgrade to Medium",
    ]
    try:
        body = page.inner_text("body")[:24000]
    except Exception:
        body = ""
    return any(m in body for m in markers)


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
    def normalize_code_text(text: str) -> str:
        # Keep indentation meaningful while removing common extraction artifacts.
        text = (
            text.replace("\r\n", "\n")
            .replace("\r", "\n")
            .replace("\u00a0", " ")  # nbsp -> space
            .replace("\u200b", "")   # zero-width space
        )
        lines = text.split("\n")

        # Heuristic: some embeds become "line + empty line + line + empty line".
        if len(lines) >= 6:
            odd = lines[1::2]
            even = lines[0::2]
            odd_blank_ratio = (
                sum(1 for ln in odd if ln.strip() == "") / max(1, len(odd))
            )
            even_nonblank_ratio = (
                sum(1 for ln in even if ln.strip() != "") / max(1, len(even))
            )
            if odd_blank_ratio > 0.85 and even_nonblank_ratio > 0.6:
                lines = even

        # Trim right side only; preserve leading spaces for Python indentation.
        lines = [ln.rstrip(" \t") for ln in lines]

        # Heuristic for certain GitHub embeds: indentation level collapses to 1 space.
        # If most indented lines start with exactly one leading space, expand it to 4.
        indented = [ln for ln in lines if ln.startswith(" ") and ln.strip()]
        if indented:
            one_space_ratio = (
                sum(1 for ln in indented if len(ln) > 1 and ln[0] == " " and (len(ln) == 1 or ln[1] != " "))
                / len(indented)
            )
            if one_space_ratio > 0.75:
                lines = [("    " + ln[1:]) if (ln.startswith(" ") and (len(ln) == 1 or ln[1] != " ")) else ln for ln in lines]

        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text).strip("\n")
        return text

    code_node = pre.find("code") or pre

    # 1) Prefer explicit line wrappers used by many syntax highlighters.
    line_nodes = code_node.select(
        ".line, .code-line, .view-line, [data-line], [data-code-line], [class*='line-']"
    )
    if len(line_nodes) >= 2:
        lines = [ln.get_text("", strip=False).rstrip("\n\r") for ln in line_nodes]
        text = "\n".join(lines)
        text = normalize_code_text(text)
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
    text = normalize_code_text(text)
    if text:
        return text

    # 3) Fallback.
    return normalize_code_text(code_node.get_text("", strip=False))


def render_code_block(code_text: str, code_mode: str = "fenced", lang: str = "") -> str:
    if code_mode == "html":
        esc = (
            code_text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        class_attr = f' class="language-{lang}"' if lang else ""
        return f"<pre><code{class_attr}>{esc}</code></pre>"
    return f"```{lang}\n{code_text}\n```"


def extract_embed_code_from_html(html: str, retries: int = 3) -> str:
    """Best-effort extraction for code embeds (gist/medium media/etc.)."""
    s = BeautifulSoup(html, "html.parser")

    # 0) Prefer raw-source links when present (best fidelity for indentation).
    raw_candidates: List[str] = []
    for a in s.find_all("a"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        txt = a.get_text(" ", strip=True).lower()
        h = href.lower()
        if (
            "raw.githubusercontent.com" in h
            or "/raw/" in h
            or ("gist.github.com" in h and "raw" in h)
            or txt == "view raw"
            or "view raw" in txt
        ):
            raw_candidates.append(href)

    seen_raw: set[str] = set()
    for href in raw_candidates:
        if href in seen_raw:
            continue
        seen_raw.add(href)
        resp = fetch_with_retries(href, retries=retries, timeout_s=20)
        if resp is None:
            continue
        ctype = (resp.headers.get("content-type") or "").lower()
        text = resp.text.replace("\r\n", "\n").replace("\r", "\n")
        # Skip HTML wrappers unless explicitly text/plain.
        if "<html" in text[:300].lower() and "text/plain" not in ctype:
            continue
        if text.count("\n") >= 3:
            return text.strip("\n")

    # 1) Standard code blocks.
    chunks: List[str] = []
    for pre in s.find_all("pre"):
        txt = extract_code_text(pre)
        if txt:
            chunks.append(txt)
    if chunks:
        return "\n\n".join(chunks).strip()

    # 2) GitHub/Gist-like line wrappers.
    line_nodes = s.select(".blob-code, .js-file-line, .line, [data-line-number]")
    if len(line_nodes) >= 2:
        lines = [
            ln.get_text("", strip=False)
            .replace("\u00a0", " ")
            .replace("\u200b", "")
            .rstrip("\n\r")
            for ln in line_nodes
        ]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text).strip("\n")
        if text:
            return text

    return ""


def preserve_math_and_code(
    soup: BeautifulSoup,
    code_mode: str = "fenced",
    page_url: str | None = None,
    retries: int = 3,
) -> Tuple[BeautifulSoup, List[Tuple[str, str]]]:
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

    # Handle embedded code iframes (e.g. medium.com/media/* embeds from GitHub gist).
    for iframe in soup.find_all("iframe"):
        raw_src = iframe.get("src")
        if not raw_src:
            iframe.decompose()
            continue

        src = urljoin(page_url, raw_src) if page_url else raw_src
        title = (iframe.get("title") or "Embedded content").strip()
        md = ""

        # Try to fetch and inline actual code content.
        resp = fetch_with_retries(src, retries=retries, timeout_s=20)
        if resp is not None and resp.text:
            embed_code = extract_embed_code_from_html(resp.text, retries=retries)
            if embed_code:
                heading = f"**{title}**\n\n" if title else ""
                md = heading + render_code_block(embed_code, code_mode=code_mode)

        # Fallback: keep a link so embeds are never silently dropped.
        if not md:
            md = f"[{title}]({src})"

        iframe.replace_with(add_placeholder(md))

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

        md = render_code_block(code_text, code_mode=code_mode, lang=lang)
        pre.replace_with(add_placeholder(md))

    return soup, placeholders


def to_markdown(
    html: str,
    code_mode: str = "fenced",
    page_url: str | None = None,
    retries: int = 3,
) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["style", "noscript"]):
        tag.decompose()

    soup, placeholders = preserve_math_and_code(
        soup,
        code_mode=code_mode,
        page_url=page_url,
        retries=retries,
    )

    # Any unprocessed iframes should not leak into markdown output.
    for iframe in soup.find_all("iframe"):
        iframe.decompose()

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


def hydrate_page_content(page: Page, idle_timeout_ms: int) -> None:
    """Best-effort hydration for lazy content (images/embeds/code cards)."""
    dismiss_medium_popups(page)
    try_wait_network_idle(page, timeout_ms=idle_timeout_ms)
    try:
        page.evaluate(
            """
            async () => {
              let prev = -1;
              for (let i = 0; i < 16; i++) {
                const h = document.body ? document.body.scrollHeight : 0;
                window.scrollBy(0, Math.max(800, window.innerHeight * 0.9));
                await new Promise((r) => setTimeout(r, 260));
                if (h === prev) {
                  break;
                }
                prev = h;
              }
              window.scrollTo(0, 0);
            }
            """
        )
    except Exception:
        pass
    try_wait_network_idle(page, timeout_ms=min(idle_timeout_ms, 4000))
    dismiss_medium_popups(page)


def expand_collapsed_content(page: Page) -> None:
    """Best-effort click on content expanders before extraction."""
    selectors = [
        "button:has-text('Read more')",
        "button:has-text('Show more')",
        "button:has-text('See more')",
        "button:has-text('Continue reading')",
        "button:has-text('View more')",
        "a:has-text('Read more')",
        "a:has-text('Continue reading')",
        "[role='button']:has-text('Read more')",
        "[role='button']:has-text('Show more')",
    ]
    for _ in range(3):
        for sel in selectors:
            try:
                loc = page.locator(sel)
                n = min(loc.count(), 8)
                for i in range(n):
                    item = loc.nth(i)
                    if item.is_visible():
                        item.click(timeout=800)
                        page.wait_for_timeout(180)
            except Exception:
                pass


def pick_page_from_context(context, target_url: str, use_existing_page: bool) -> Page:
    pages = context.pages
    if use_existing_page and pages:
        target_canon = canonical_page_url(target_url)

        # Prefer exact article-tab match.
        for p in reversed(pages):
            try:
                purl = p.url or ""
            except Exception:
                purl = ""
            if canonical_page_url(purl) == target_canon:
                return p

        # Fallback: same host.
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


def inline_iframe_code_embeds(page: Page, article_html: str, retries: int, debug_embeds: bool = False) -> str:
    """Resolve code embeds rendered inside iframes into inline <pre><code> blocks.

    Medium often stores GitHub/Gist snippets as iframes (medium.com/media/*).
    If we drop iframes before markdown conversion, code gets lost. This helper
    opens each iframe URL with Playwright (JS-rendered) and inlines extracted code.
    """
    soup = BeautifulSoup(article_html, "html.parser")
    iframes = soup.find_all("iframe")
    if not iframes:
        return article_html

    total = len(iframes)
    if debug_embeds:
        print(f"[embed] found {total} iframe(s) in article", file=sys.stderr)

    for idx, iframe in enumerate(iframes, start=1):
        raw_src = iframe.get("src")
        if not raw_src:
            if debug_embeds:
                print(f"[embed {idx}/{total}] skip: missing src", file=sys.stderr)
            continue

        src = urljoin(page.url, raw_src)
        title = (iframe.get("title") or "").strip()
        embed_code = ""
        attempts = 0
        last_error = ""
        display_title = title if title else "(untitled)"

        for i in range(retries + 1):
            attempts += 1
            embed_page = None
            try:
                embed_page = page.context.new_page()
                embed_page.goto(src, wait_until="domcontentloaded", timeout=25000)
                embed_page.wait_for_timeout(1200)
                # Give script-based embeds a chance to render.
                try:
                    embed_page.wait_for_selector(
                        "pre, code, .blob-code, .js-file-line, [data-line-number]",
                        timeout=3500,
                    )
                except Exception:
                    pass
                embed_html = embed_page.content()
                embed_code = extract_embed_code_from_html(embed_html, retries=retries)
                if embed_code:
                    break
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                if i == retries:
                    break
            finally:
                if embed_page is not None:
                    try:
                        embed_page.close()
                    except Exception:
                        pass
                if not embed_code and i < retries:
                    time.sleep(min(4, 1.2 * (2**i)))

        if not embed_code:
            if debug_embeds:
                reason = last_error if last_error else "no code nodes found after render"
                print(
                    f"[embed {idx}/{total}] failed: title={display_title!r}, src={src}, attempts={attempts}, reason={reason}",
                    file=sys.stderr,
                )
            continue

        if debug_embeds:
            line_count = len(embed_code.splitlines())
            print(
                f"[embed {idx}/{total}] success: title={display_title!r}, src={src}, attempts={attempts}, lines={line_count}",
                file=sys.stderr,
            )

        container = soup.new_tag("div")
        if title:
            title_tag = soup.new_tag("p")
            strong = soup.new_tag("strong")
            strong.string = title
            title_tag.append(strong)
            container.append(title_tag)

        pre = soup.new_tag("pre")
        code = soup.new_tag("code")
        code.string = embed_code
        pre.append(code)
        container.append(pre)
        iframe.replace_with(container)

    return str(soup)


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
    fail_on_member_preview: bool,
    debug_embeds: bool = False,
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
        elif canonical_page_url(page.url or "") != canonical_page_url(url):
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
        else:
            # Even with --use-existing-page, ensure lazy embeds/code cards are loaded.
            hydrate_page_content(page, idle_timeout_ms=idle_timeout_ms)

        expand_collapsed_content(page)
        hydrate_page_content(page, idle_timeout_ms=idle_timeout_ms)

        if looks_like_member_preview(page):
            msg = (
                "Detected Medium member-only/preview signals. "
                "You may be logged in but still lack full access for this story. "
                "Export will include only currently visible content."
            )
            if fail_on_member_preview:
                raise RuntimeError(msg)
            print(f"[warn] {msg}", file=sys.stderr)

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

        if article_html.strip():
            try:
                article_html = inline_iframe_code_embeds(
                    page,
                    article_html,
                    retries=max(0, retries),
                    debug_embeds=debug_embeds,
                )
            except Exception:
                # Keep original article content if embed expansion fails.
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
    fail_on_member_preview: bool,
    export_html: bool,
    code_mode: str,
    debug_embeds: bool = False,
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
        fail_on_member_preview=fail_on_member_preview,
        debug_embeds=debug_embeds,
    )

    safe_slug = slugify(filename if filename else title)
    md_path = out_dir / f"{safe_slug}.md"
    assets_dir = out_dir / f"{safe_slug}_assets"

    soup = BeautifulSoup(article_html, "html.parser")
    download_images(soup, url, assets_dir, retries=retries)
    html_content = str(soup)

    body_md = to_markdown(
        html_content,
        code_mode=code_mode,
        page_url=url,
        retries=retries,
    )
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
        "--fail-on-member-preview",
        action="store_true",
        help="Fail fast when page appears to be Medium member-only preview/paywall state.",
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Show browser window (disable headless mode)",
    )
    parser.add_argument(
        "--debug-embeds",
        action="store_true",
        help="Print debug logs for iframe code embeds: success/failure, attempts, and extracted line count.",
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
        fail_on_member_preview=bool(args.fail_on_member_preview),
        export_html=bool(args.export_html),
        code_mode=args.code_mode,
        debug_embeds=bool(args.debug_embeds),
    )
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
