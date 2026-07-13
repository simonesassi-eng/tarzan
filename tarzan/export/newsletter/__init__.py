"""Newsletter package — HTML email digest from a PortfolioMetrics.

Split from a single 3,500-line module into cohesive submodules
(_constants, _format, _charts, _sections_alloc, _sections_perf) with this
package __init__ as the orchestrator + public API. Import surface is
unchanged: ``from tarzan.export.newsletter import render_newsletter`` etc.
still work, and the _perf_series re-exports keep their object identity for
the audit identity test.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from tarzan import runtime as _runtime
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics

from tarzan.export.newsletter._constants import (  # noqa: F401 (re-exported)
    ASSET_BG,
    ASSET_COLORS,
    ASSET_CLASS_ORDER,
    GEO_COLORS,
    MARKET_REGION_COLORS,
    PALETTE,
    _NEWSLETTER_CLASS_ORDER,
    _NewsletterContext,
    _PERF_CLASS_ORDER,
)
from tarzan.export.newsletter._format import _colorize_pct
from tarzan.export.newsletter import _charts as _charts_mod
from tarzan.export.newsletter._sections_alloc import (
    _build_headline,
    _build_header,
    _build_hero,
    _build_tax_note,
    _build_methodology,
    _build_diversification,
    _build_holdings,
    _build_optimizer,
    _build_return_contrib,
    _build_preheader,
)
from tarzan.export.newsletter._sections_perf import (
    _build_performance30,
    _build_movers,
    _build_returns_snapshot,
    _build_performance,
    _build_markets,
    _build_risk_profile,
)

# Preserve the _perf_series re-export surface BY IDENTITY (test_audit asserts
# ``newsletter.<name> is _perf_series.<name>``).
from tarzan.export._perf_series import (  # noqa: F401
    _flow_list,
    _geo_benchmark_series,
    _norm_series,
    _perf_full_series,
    _perf_level_series,
    _perf_vol_series,
    _perf_window,
    _window_money_pnl,
    _window_twror,
    market_snapshot,
)

logger = logging.getLogger(__name__)

def build_context(
    metrics: PortfolioMetrics,
    config: InvestorConfig,
    issue_number: int = 1,
    benchmark_alpha_beta: str = "S&P 500",
    benchmark_geo: str = "MSCI ACWI",
    ai_summary: Optional[str] = None,
) -> dict[str, Any]:
    """Build the full Jinja2 context dict for the newsletter template.

    Args:
        metrics: Computed portfolio metrics.
        config: Investor configuration.
        issue_number: Sequential issue number for branding.
        benchmark_alpha_beta: Display name of α/β benchmark (from constants.yaml).
        benchmark_geo: Display name of geographic allocation benchmark.

    Returns:
        A dict with all keys consumed by ``portfolio_digest.html.j2``.
    """
    # Reset the per-render SVG clipPath id counters so a render's element ids
    # depend only on how many charts it draws, not on how many newsletters the
    # process rendered before it. Without this, two renders in one process
    # emit different ids (dg1 vs dg2, ...), making the HTML non-reproducible —
    # which defeats deterministic mode. Ids are internal references (visually
    # invisible) and each render is a standalone document, so resetting is safe
    # and never collides.
    _charts_mod.reset_spark_uids()
    nctx = _NewsletterContext(
        metrics=metrics,
        config=config,
        issue_number=issue_number,
        benchmark_alpha_beta=benchmark_alpha_beta,
        benchmark_geo=benchmark_geo,
    )
    hero = _build_hero(nctx)
    return {
        "palette": PALETTE,
        "header": _build_header(nctx),
        "headline": _build_headline(nctx, hero),
        "hero": hero,
        "performance30": _build_performance30(nctx),
        "ai_summary": ai_summary,
        "ai_summary_html": _colorize_pct(ai_summary) if ai_summary else None,
        "movers": _build_movers(nctx),
        "diversification": _build_diversification(nctx),
        "holdings": _build_holdings(nctx),
        "returns_snapshot": _build_returns_snapshot(nctx),
        "performance": _build_performance(nctx),
        "markets": _build_markets(nctx),
        "risk_profile": _build_risk_profile(nctx),
        "optimizer": _build_optimizer(nctx),
        "return_contrib": _build_return_contrib(nctx),
        "tax_note": _build_tax_note(nctx),
        "methodology": _build_methodology(nctx),
        "preheader": _build_preheader(nctx, hero),
        "footer": {
            # Pinned stamp in deterministic mode so the header does not vary
            # run-to-run (live now() otherwise).
            "generated_at": _runtime.now_stamp("%d %b %Y, %H:%M"),
            "version": "v2.0",
        },
    }

def render_newsletter(
    metrics: PortfolioMetrics,
    config: InvestorConfig,
    issue_number: int = 1,
    benchmark_alpha_beta: Optional[str] = None,
    benchmark_geo: Optional[str] = None,
    ai_summary: Optional[str] = None,
) -> str:
    """Render the newsletter HTML to a string.

    Args:
        metrics: Computed portfolio metrics.
        config: Investor configuration.
        issue_number: Sequential issue number for branding.
        benchmark_alpha_beta: Display name of α/β benchmark. When None it is
            resolved from configuration (instrument_taxonomy.csv
            ``is_benchmark_alpha_beta``) so the label/tag always match the
            benchmark the engine actually computed α/β against.
        benchmark_geo: Display name of geographic allocation benchmark. When
            None it is resolved from configuration
            (``is_benchmark_geo``).

    Returns:
        The full HTML newsletter as a single string.
    """
    # Resolve benchmark display names from config when not explicitly passed.
    # This closes a mismatch where the α/β footnote/tag could name a different
    # index than the β=1.00 row the engine produced.
    from tarzan import config as _cfg
    if benchmark_alpha_beta is None:
        try:
            benchmark_alpha_beta = _cfg.benchmark_beta_name()
        except Exception:
            benchmark_alpha_beta = "S&P 500"
    if benchmark_geo is None:
        try:
            benchmark_geo = _cfg.benchmark_geo_allocation()
        except Exception:
            benchmark_geo = "MSCI ACWI"

    # Resolve the optional AI market-context summary when not explicitly passed,
    # so every caller (CLI + email) renders the SAME newsletter. generate_summary
    # is fully self-guarding: it returns None when no GEMINI_API_KEY is set, in a
    # deterministic run, or on any error — it never raises and never blocks the
    # render. Pass ai_summary="" to force the block off even when a key exists.
    if ai_summary is None:
        from tarzan.export.ai_summary import generate_summary
        ai_summary = generate_summary(metrics, config)

    template_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "templates")
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("portfolio_digest.html.j2")
    context = build_context(
        metrics, config, issue_number, benchmark_alpha_beta, benchmark_geo,
        ai_summary=ai_summary,
    )
    return template.render(**context)

def generate_newsletter(
    metrics: PortfolioMetrics,
    config: InvestorConfig,
    output_dir: str,
    issue_number: int = 1,
    benchmark_alpha_beta: Optional[str] = None,
    benchmark_geo: Optional[str] = None,
) -> str:
    """Render the newsletter and write it to disk.

    Writes ``portfolio_digest_<YYYYMMDD_HHMM>.html`` into ``output_dir``. The
    rendering goes through :func:`render_newsletter`, so a CLI run and an
    emailed send produce the same HTML (benchmark names and the optional AI
    market-context summary are resolved there).

    Args:
        metrics: Computed portfolio metrics.
        config: Investor configuration.
        output_dir: Directory for the output file.
        issue_number: Sequential issue number for branding.
        benchmark_alpha_beta: Display name of α/β benchmark. When None,
            resolved from configuration (so labels match the engine).
        benchmark_geo: Display name of geographic allocation benchmark.
            When None, resolved from configuration.

    Returns:
        Path to the generated HTML file.
    """
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    filepath = os.path.join(output_dir, f"portfolio_digest_{date_str}.html")
    html = render_newsletter(
        metrics, config, issue_number, benchmark_alpha_beta, benchmark_geo,
    )
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Newsletter written to %s", filepath)
    return filepath

