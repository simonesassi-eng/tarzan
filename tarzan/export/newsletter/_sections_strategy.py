"""Strategy section: the investor's own written thesis, read from a text file.

Content lives in ``input/strategy.txt`` (personal data, gitignored) — plain text
the user edits freely, NOT generated or paraphrased here. This module only parses
the lightweight block format and renders it, so the newsletter states the strategy
in the investor's own words rather than a machine summary.

Format::

    # comment line (dropped)
    ## Block heading
    body paragraph, blank-line separated

Absent/empty file → ``{"available": False}`` and the section disappears.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from tarzan.export.newsletter._constants import PALETTE, _NewsletterContext

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
STRATEGY_FILE = ROOT / "input" / "strategy.txt"


def parse_strategy(text: str) -> list[tuple[str, list[str]]]:
    """Parse the block format into ``[(heading, [paragraph, ...]), ...]``.

    Lines starting with ``##`` open a block; ``#`` alone is a comment. Text
    before the first heading is kept under an empty heading so a file with no
    headings still renders.
    """
    blocks: list[tuple[str, list[str]]] = []
    heading = ""
    lines: list[str] = []

    def flush():
        # Group consecutive non-blank lines into paragraphs. A line ending in a
        # full stop closes its paragraph: that keeps a one-sentence-per-line
        # list (the "five building blocks") as separate readable items instead
        # of merging it into one wall of text, while ordinary wrapped prose
        # (lines broken mid-sentence) still joins up.
        paras, cur = [], []
        for ln in lines:
            s = ln.strip()
            if s:
                cur.append(s)
                if s.endswith((".", ":", "!", "?")):
                    paras.append(" ".join(cur)); cur = []
            elif cur:
                paras.append(" ".join(cur)); cur = []
        if cur:
            paras.append(" ".join(cur))
        if paras:
            blocks.append((heading, paras))

    for raw in text.splitlines():
        if raw.startswith("##"):
            flush()
            heading = raw.lstrip("#").strip()
            lines = []
        elif raw.startswith("#"):
            continue                      # comment
        else:
            lines.append(raw)
    flush()
    return blocks


def _build_strategy(ctx: _NewsletterContext) -> dict:
    """Render ``input/strategy.txt`` as the newsletter's strategy section.

    Skipped under pytest: the note is personal free text and the golden-HTML
    fixture is a COMMITTED file, so letting it in would publish the investor's
    holdings and thesis into the repository.
    """
    try:
        if "PYTEST_CURRENT_TEST" in os.environ:
            return {"available": False, "html": ""}
        if not STRATEGY_FILE.exists():
            return {"available": False, "html": ""}
        text = STRATEGY_FILE.read_text(encoding="utf-8")
    except OSError as e:                  # unreadable file must never break the render
        logger.warning("Strategy note skipped (%s): %s", type(e).__name__, e)
        return {"available": False, "html": ""}

    blocks = parse_strategy(text)
    if not blocks:
        return {"available": False, "html": ""}

    P = PALETTE
    parts: list[str] = []
    for heading, paras in blocks:
        body = "".join(
            f'<div style="margin-top:{6 if i else 0}px;font-size:12.5px;'
            f'color:{P["muted"]};line-height:1.65;">{p}</div>'
            for i, p in enumerate(paras)
        )
        head = (f'<div style="font-size:11px;font-weight:700;letter-spacing:0.06em;'
                f'color:{P["accent"]};text-transform:uppercase;">{heading}</div>'
                if heading else "")
        parts.append(
            f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
            f'border="0" style="margin-top:10px;background:{P["card_alt"]};'
            f'border:1px solid {P["border"]};border-left:3px solid {P["accent"]};'
            f'border-radius:8px;border-collapse:separate;border-spacing:0;">'
            f'<tr><td style="padding:12px 14px;">{head}{body}</td></tr></table>'
        )

    return {
        "available": True,
        "sub": "Why this portfolio is built the way it is, in your own words, "
               "from input/strategy.txt.",
        "html": "".join(parts),
    }
