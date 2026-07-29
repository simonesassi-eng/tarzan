"""Pre-delivery semantic invariants for the portfolio newsletter.

The renderer is allowed to format canonical analytical data, but it is not
allowed to resolve another market symbol or relabel a chart with a return from
a different window.  This module verifies those rules on the completed render
before any durable delivery claim or SMTP invocation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping


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
    if resolution_errors:
        errors.append(
            "benchmark preprocessing did not resolve the complete universe: "
            + "; ".join(resolution_errors)
        )
    if "_preprocess_benchmarks" in degraded:
        errors.append("benchmark preprocessing computer failed")
    if "_live_1d" in degraded:
        errors.append("intraday preprocessing source validation failed")
    if selected_names != expected_names:
        missing = sorted(expected_names - selected_names)
        extra = sorted(selected_names - expected_names)
        errors.append(
            f"benchmark ticker catalog differs from configuration "
            f"(missing={missing!r}, extra={extra!r})"
        )
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

    expected_intraday: set[str] = set()
    for frame in (hp, getattr(metrics, "holdings_df", None)):
        if frame is None or getattr(frame, "empty", True) or "ticker" not in frame.columns:
            continue
        expected_intraday.update(
            _text(ticker) for ticker in frame["ticker"].dropna() if _text(ticker)
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

    # Recompute the 30-day endpoint independently from the raw metrics, then
    # compare it with both the renderer's endpoint and the visible rounded
    # label. This catches a label sourced from a generic 1M bucket while the
    # line ends on a different shared close.
    if getattr(metrics, "actual_value_series", None) is not None:
        from tarzan.export._perf_series import _perf_window

        geo_name = cfg.benchmark_geo_allocation()
        expected_window = _perf_window(metrics, 30, geo_name)
        perf_audit = audit.get("performance_30d", {})
        if expected_window is None:
            errors.append("30-day performance window is unavailable")
        elif not isinstance(perf_audit, Mapping):
            errors.append("30-day performance render audit is missing")
        else:
            expected_endpoints = expected_window.get("endpoints", {})
            rendered_endpoints = perf_audit.get("endpoints", {})
            legend_values = perf_audit.get("legend_values", {})
            legend_labels = perf_audit.get("legend_labels", {})
            # Every line the 30-day chart can draw is audited here. A line
            # absent from this tuple renders unverified, which is the one
            # failure mode the gate exists to prevent.
            for key in ("twror", "pnl_pct", "unreal_pct", "acwi"):
                expected_value = expected_endpoints.get(key)
                if expected_value is None:
                    continue
                rendered_value = rendered_endpoints.get(key)
                label_value = legend_values.get(key)
                label = _text(legend_labels.get(key))
                if rendered_value is None or not math.isclose(
                    float(rendered_value), float(expected_value), abs_tol=1e-9
                ):
                    errors.append(
                        f"30-day {key} line endpoint differs from the shared-close endpoint"
                    )
                if label_value is None or not math.isclose(
                    float(label_value), float(expected_value), abs_tol=1e-9
                ):
                    errors.append(f"30-day {key} legend uses a different endpoint")
                displayed = _displayed_percent(label)
                if displayed is None or not math.isclose(
                    displayed, float(expected_value), abs_tol=0.0051
                ):
                    errors.append(
                        f"30-day {key} visible label {label!r} disagrees with "
                        f"endpoint {float(expected_value):+.6f}%"
                    )
                if label and label not in newsletter_html:
                    errors.append(f"30-day {key} audited label is absent from rendered HTML")

    return tuple(errors)
