"""Pre-delivery semantic invariants for the portfolio newsletter.

The renderer is allowed to format canonical analytical data, but it is not
allowed to resolve another market symbol or relabel a chart with a return from
a different window.  This module verifies those rules on the completed render
before any durable delivery claim or SMTP invocation.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping

logger = logging.getLogger("tarzan.newsletter")


# The sign is OPTIONAL. ``_pct(..., signed=True)`` deliberately omits it when
# the value rounds to zero (``_sign_for`` returns "" so a residual \u22120.004%
# prints as "0.00%" rather than as a signed zero reading like a real move
# down). Requiring a sign here made that correct rendering unparseable, and an
# unparseable label is reported as disagreeing with its endpoint \u2014 blocking
# delivery over a line that was drawn and labelled correctly.
_DISPLAYED_PERCENT_RE = re.compile(r"([+\-\u2212]?)(\d+(?:\.\d+)?)%$")


def _text(value: object) -> str:
    return str(value or "").strip()


def _identity_map(frame, name_column: str, ticker_column: str) -> dict[str, set[str]]:
    """Project a frame into ``name -> exact ticker set`` without coercing NaN."""
    if frame is None or getattr(frame, "empty", True):
        return {}
    if name_column not in frame.columns or ticker_column not in frame.columns:
        return {}
    identities: dict[str, set[str]] = {}
    for _, row in frame.iterrows():
        name = _text(row.get(name_column))
        ticker = _text(row.get(ticker_column))
        if name:
            identities.setdefault(name, set()).add(ticker)
    return identities


def _check_projection(
    errors: list[str],
    projection: str,
    actual: Mapping[str, set[str]],
    expected: Mapping[str, str],
) -> None:
    for name, ticker in expected.items():
        seen = actual.get(name)
        if seen is None:
            errors.append(f"{projection} is missing benchmark {name}")
        elif seen != {ticker}:
            errors.append(
                f"{projection} uses {sorted(seen)!r} for {name}; expected {ticker!r}"
            )


def _displayed_percent(label: str) -> float | None:
    match = _DISPLAYED_PERCENT_RE.search(label)
    if match is None:
        return None
    value = float(match.group(2))
    return -value if match.group(1) in {"-", "\u2212"} else value


def validate_newsletter_semantics(
    metrics,
    semantic_audit: Mapping[str, object] | None,
    newsletter_html: str,
) -> tuple[str, ...]:
    """Return every identity/window violation that must block delivery.

    Empty legacy/test metrics with no benchmark contract are ignored. A real
    MetricsEngine result always exposes either resolved benchmark tickers,
    explicit resolution errors, or a degraded preprocessing computer.
    """
    resolution_errors = tuple(
        _text(error)
        for error in (getattr(metrics, "benchmark_resolution_errors", ()) or ())
        if _text(error)
    )
    selected = {
        _text(name): _text(ticker)
        for name, ticker in (getattr(metrics, "benchmark_tickers", {}) or {}).items()
        if _text(name)
    }
    degraded = set(getattr(metrics, "degraded_computers", ()) or ())
    hp = getattr(metrics, "holding_performance", None)
    comparison = getattr(metrics, "benchmark_comparison", None)
    histories = getattr(metrics, "benchmark_histories", {}) or {}
    historical_risk = getattr(metrics, "historical_risk", None) or {}
    has_contract = bool(
        selected
        or resolution_errors
        or "_preprocess_benchmarks" in degraded
        or histories
        or not getattr(comparison, "empty", True)
    )
    if not has_contract:
        return ()

    from tarzan import config as cfg

    errors: list[str] = []
    definitions = {_text(name): _text(ticker) for name, ticker in cfg.benchmarks().items()}
    expected_names = set(definitions)
    selected_names = set(selected)

    # Only the benchmarks the portfolio is actually MEASURED against — the geo
    # and alpha/beta references drawn in the "vs market" chart — are required.
    # The other ~50 configured names are tracked for reference; a transient
    # Yahoo outage on one of them (X25E returned no history on 20 Aug 2026 under
    # runner throttling) must drop that one row, NOT block the whole digest —
    # three sends were lost to exactly this. Availability degrades; identity is
    # still fully enforced below on whatever DID resolve.
    critical = {_text(cfg.benchmark_geo_allocation()), _text(cfg.benchmark_beta_name())}
    critical.discard("")
    unresolved = expected_names - selected_names
    missing_critical = sorted(unresolved & critical)
    if missing_critical:
        errors.append(
            f"critical benchmark(s) did not resolve: {missing_critical!r}"
        )
    dropped = sorted(unresolved - critical)
    if dropped:
        logger.warning(
            "Benchmark(s) unavailable this run, dropped from the digest (not a "
            "delivery blocker): %s",
            dropped,
        )
    extra = sorted(selected_names - expected_names)
    if extra:
        errors.append(
            f"benchmark ticker catalog has unexpected entries: extra={extra!r}"
        )
    # A resolution error blocks only when it concerns a CRITICAL benchmark.
    critical_resolution_errors = [
        error for error in resolution_errors
        if any(name and name in error for name in critical)
    ]
    if critical_resolution_errors:
        errors.append(
            "critical benchmark preprocessing did not resolve: "
            + "; ".join(critical_resolution_errors)
        )
    if "_preprocess_benchmarks" in degraded:
        errors.append("benchmark preprocessing computer failed")
    if "_live_1d" in degraded:
        errors.append("intraday preprocessing source validation failed")
    for name, ticker in selected.items():
        if not ticker:
            errors.append(f"benchmark {name} has an empty provider ticker")

    benchmark_rows = hp
    if hp is not None and not getattr(hp, "empty", True) and "type" in hp.columns:
        benchmark_rows = hp[
            hp["type"].astype(str).str.contains("enchmark", case=False, na=False)
        ]
    _check_projection(
        errors,
        "holding_performance",
        _identity_map(benchmark_rows, "name", "ticker"),
        selected,
    )
    _check_projection(
        errors,
        "benchmark_comparison",
        _identity_map(comparison, "benchmark", "ticker"),
        selected,
    )

    risk_rows: dict[str, set[str]] = {}
    for row in historical_risk.get("instruments", ()) or ():
        name = _text(row.get("label"))
        if name:
            risk_rows.setdefault(name, set()).add(_text(row.get("ticker")))
    _check_projection(errors, "historical_risk", risk_rows, selected)

    for name, ticker in selected.items():
        series = histories.get(name)
        if series is None:
            errors.append(f"benchmark_histories is missing benchmark {name}")
            continue
        series_name = _text(getattr(series, "name", ""))
        resolved = _text(getattr(series, "attrs", {}).get("resolved_ticker"))
        requested = _text(getattr(series, "attrs", {}).get("requested_ticker"))
        if series_name != ticker:
            errors.append(
                f"benchmark_histories[{name!r}].name is {series_name!r}; expected {ticker!r}"
            )
        if resolved != ticker:
            errors.append(
                f"benchmark_histories[{name!r}] resolved ticker is {resolved!r}; "
                f"expected {ticker!r}"
            )
        if requested != definitions.get(name):
            errors.append(
                f"benchmark_histories[{name!r}] provenance is {requested!r}; "
                f"expected input {definitions.get(name)!r}"
            )

    # History/current must use the canonical identity. Intraday is a separate
    # data capability: a guarded same-root EUR sibling is allowed only when the
    # preprocessing provenance explicitly records that fallback.
    from tarzan.data.market_quotes import _sibling_symbols

    for record in getattr(metrics, "ticker_resolutions", ()) or ():
        canonical = _text(record.get("canonical_ticker"))
        status = _text(record.get("status"))
        if status == "CONFLICT":
            errors.append(
                f"holding ticker resolution conflict for "
                f"{_text(record.get('name')) or canonical}"
            )
            continue
        if not canonical:
            continue
        for field in ("history_ticker", "current_ticker"):
            value = _text(record.get(field))
            if not value:
                continue
            identities = {
                part.strip() for part in value.split(" / ") if part.strip()
            }
            if identities != {canonical}:
                errors.append(
                    f"holding {canonical} has alternate "
                    f"{field}={sorted(identities)!r}"
                )

        intraday_value = _text(record.get("intraday_effective_ticker"))
        intraday_reason = _text(record.get("intraday_reason"))
        intraday_identities = {
            part.strip()
            for part in intraday_value.split(" / ")
            if part.strip()
        }
        allowed_intraday = {canonical, *_sibling_symbols(canonical)}
        invalid_intraday = intraday_identities - allowed_intraday
        if invalid_intraday:
            errors.append(
                f"holding {canonical} has invalid intraday source(s) "
                f"{sorted(invalid_intraday)!r}"
            )
        if (
            any(ticker != canonical for ticker in intraday_identities)
            and "sibling fallback" not in intraday_reason.casefold()
        ):
            errors.append(
                f"holding {canonical} has an undocumented intraday fallback"
            )

    audit = semantic_audit or {}
    intraday = audit.get("performance_intraday", {})
    if not isinstance(intraday, Mapping):
        intraday = {}
    origin = _text(intraday.get("origin"))
    requested = tuple(
        _text(ticker)
        for ticker in intraday.get("requested_tickers", ())
        if _text(ticker)
    )
    returned = tuple(
        _text(ticker)
        for ticker in intraday.get("returned_tickers", ())
        if _text(ticker)
    )
    rendered_sources_raw = intraday.get("source_tickers", {})
    rendered_sources = (
        {
            _text(canonical): _text(source)
            for canonical, source in rendered_sources_raw.items()
            if _text(canonical)
        }
        if isinstance(rendered_sources_raw, Mapping)
        else {}
    )
    preprocessed_requested = tuple(
        _text(ticker)
        for ticker in (
            getattr(metrics, "intraday_requested_tickers", ()) or ()
        )
        if _text(ticker)
    )
    preprocessed_quotes = dict(
        getattr(metrics, "intraday_quotes", {}) or {}
    )

    # The intraday request is built from ``holding_performance``, which carries
    # only holdings with a usable price history (>= 2 closes) and drops those
    # whose order mechanics are unavailable. ``holdings_df`` keeps every
    # valuation-accepted holding, history or not — so unioning the two demands
    # intraday for tickers the request set structurally cannot contain, and a
    # freshly-bought or thinly-listed instrument blocks delivery on its first
    # run. The analytical ticker set for THIS check is the performance frame.
    expected_intraday: set[str] = set()
    hp_frame = hp
    if hp_frame is None or getattr(hp_frame, "empty", True) or "ticker" not in getattr(hp_frame, "columns", ()):
        # No performance frame at all: fall back to the holdings snapshot so a
        # renderer that skipped preprocessing entirely is still caught.
        hp_frame = getattr(metrics, "holdings_df", None)
    if hp_frame is not None and not getattr(hp_frame, "empty", True) and "ticker" in getattr(hp_frame, "columns", ()):
        expected_intraday.update(
            _text(ticker) for ticker in hp_frame["ticker"].dropna() if _text(ticker)
        )
    # Plus the TARGET's own sleeves, seeds included, because ``_live_1d`` requests
    # them: the 1D panel blends the target's session from their bars and refuses
    # partial coverage. On the reference book they are all in the frame anyway (every
    # target sleeve is also a tracked benchmark, and the frame carries those), so
    # this widens the expectation for a shape that is possible rather than one that
    # is present — but the request and the expectation must move together or every
    # run fails on "candidates differ", which blocks delivery.
    expected_intraday.update(
        _text(ticker)
        for ticker in (getattr(metrics, "target_weights", {}) or {})
        if _text(ticker)
    )

    if origin != "metrics_preprocessing":
        errors.append("performance intraday data did not originate in preprocessing")
    if set(preprocessed_requested) != expected_intraday:
        errors.append(
            "intraday preprocessing candidates differ from the analytical "
            f"ticker set (requested={sorted(set(preprocessed_requested))!r}, "
            f"expected={sorted(expected_intraday)!r})"
        )
    if set(requested) != set(preprocessed_requested):
        errors.append("renderer did not consume the preprocessing request set")
    if len(requested) != len(set(requested)):
        errors.append("intraday preprocessing request contains duplicate symbols")
    if set(returned) != set(preprocessed_quotes):
        errors.append("renderer intraday rows differ from the preprocessed catalog")
    if not set(returned).issubset(set(requested)):
        errors.append(
            "preprocessing returned an unrequested canonical ticker "
            f"{sorted(set(returned) - set(requested))!r}"
        )

    expected_sources: dict[str, str] = {}
    for canonical, quote in preprocessed_quotes.items():
        canonical_text = _text(canonical)
        if not isinstance(quote, Mapping):
            errors.append(
                f"preprocessed intraday quote {canonical_text} is not structured"
            )
            continue
        source = _text(
            quote.get("intraday_source_ticker")
            or quote.get("source_ticker")
            or canonical_text
        )
        expected_sources[canonical_text] = source
        if source not in {canonical_text, *_sibling_symbols(canonical_text)}:
            errors.append(
                f"preprocessed intraday quote {canonical_text} uses invalid "
                f"source {source!r}"
            )
        series = quote.get("intraday_series")
        if series is None or len(series) < 2:
            errors.append(
                f"preprocessed intraday quote {canonical_text} has no usable series"
            )
        try:
            baseline = float(quote.get("intraday_baseline"))
        except (TypeError, ValueError):
            baseline = math.nan
        if not math.isfinite(baseline) or baseline == 0:
            errors.append(
                f"preprocessed intraday quote {canonical_text} has invalid baseline"
            )
    if rendered_sources != expected_sources:
        errors.append("renderer intraday sources differ from preprocessing provenance")

    # Recompute EVERY window panel independently from the raw metrics, then
    # compare each with the renderer's endpoint and its visible rounded label.
    # This catches a label sourced from a generic return bucket while the line
    # ends on a different shared close -- the failure this gate was built for,
    # now applied to all six windows rather than only the 30-day one.
    if getattr(metrics, "actual_value_series", None) is not None:
        from tarzan.export._perf_series import _perf_window
        # 1D is the SESSION, so its window comes from the intraday bars rather than
        # from daily closes (see ``_perf_intraday_window`` for why a 1D daily window
        # is degenerate). It returns the same shape, so the loop below verifies it
        # through the identical code path as every other bucket -- the endpoint, the
        # legend value, the visible rounded label and its presence in the HTML.
        from tarzan.export.newsletter._sections_perf import _perf_intraday_window

        geo_name = cfg.benchmark_geo_allocation()
        #: The lines a window panel is DESIGNED to draw. Kept here as well as in
        #: the renderer on purpose: the check that the drawn set EQUALS the
        #: resolvable set is what stops a line from quietly disappearing, and a
        #: gate reading its expectation out of the audit it is checking would
        #: verify nothing. Total and Unrealized P&L are absent by design -- they
        #: live in the matrix, which the money-figure checks above cover.
        panel_keys = ("twror", "target", "acwi")
        window_audit = audit.get("performance_windows")
        if not isinstance(window_audit, Mapping):
            # Fall back to the single-window entry so an older audit shape is
            # reported as a missing audit rather than silently passing.
            window_audit = {"1m": audit.get("performance_30d", {})}
        for bucket in ("1d", "5d", "1m", "3m", "ytd", "1y"):
            expected_window = (
                _perf_intraday_window(metrics, geo_name) if bucket == "1d"
                else _perf_window(metrics, 30, geo_name, bucket=bucket))
            panel_audit = window_audit.get(bucket)
            if expected_window is None:
                # No session (1D) or the book does not reach back that far. Either
                # way the renderer must not have drawn a panel for it -- for 1D it
                # states the figures instead, which carry no audited line.
                if panel_audit:
                    errors.append(
                        f"{bucket} panel was rendered for an unavailable window")
                continue
            expected_endpoints = expected_window.get("endpoints", {})
            resolvable = {k for k in panel_keys
                          if expected_endpoints.get(k) is not None}
            if not resolvable:
                continue
            if bucket == "1m" and not isinstance(panel_audit, Mapping):
                errors.append("30-day performance render audit is missing")
                continue
            if not isinstance(panel_audit, Mapping):
                errors.append(f"{bucket} performance render audit is missing")
                continue
            drawn = set(panel_audit.get("drawn") or ())
            if drawn != resolvable:
                errors.append(
                    f"{bucket} panel drew {sorted(drawn)} where "
                    f"{sorted(resolvable)} resolved")
            rendered_endpoints = panel_audit.get("endpoints", {})
            legend_values = panel_audit.get("legend_values", {})
            legend_labels = panel_audit.get("legend_labels", {})
            # What the LABEL must equal is not always the drawn endpoint. A window may
            # supply its own authoritative figures — only 1D does — because a blended
            # session path is short by any sleeve the quote catalog did not return
            # while that sleeve's own 1D is known from the tape. Every other 1D cell in
            # the newsletter already labels from the tape, so this checks the
            # convention the page follows rather than the one the grid used to.
            expected_labels = dict(expected_window.get("labels") or {})
            for key in sorted(resolvable):
                expected_value = expected_endpoints.get(key)
                expected_label = expected_labels.get(key, expected_value)
                rendered_value = rendered_endpoints.get(key)
                label_value = legend_values.get(key)
                label = _text(legend_labels.get(key))
                if rendered_value is None or not math.isclose(
                    float(rendered_value), float(expected_value), abs_tol=1e-9
                ):
                    errors.append(
                        f"{bucket} {key} line endpoint differs from the "
                        f"shared-close endpoint"
                    )
                if label_value is None or not math.isclose(
                    float(label_value), float(expected_label), abs_tol=1e-9
                ):
                    errors.append(f"{bucket} {key} legend uses a different figure")
                displayed = _displayed_percent(label)
                if displayed is None or not math.isclose(
                    displayed, float(expected_label), abs_tol=0.0051
                ):
                    errors.append(
                        f"{bucket} {key} visible label {label!r} disagrees with "
                        f"{float(expected_label):+.6f}%"
                    )
                if label and label not in newsletter_html:
                    errors.append(
                        f"{bucket} {key} audited label is absent from rendered HTML")

    return tuple(errors)
