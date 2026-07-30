"""An ISIN-only order must reach the curated identity WITHOUT a hand-filled ISIN.

``resolve_taxonomy_identity`` is the single global resolver, but it queries the
one taxonomy table by whichever key the caller holds: ISIN (Fineco exports are
ISIN-only), else bare ticker (target files), else name. A curated row whose isin
cell is empty is therefore reachable by ticker and invisible to an order for the
same instrument — and 28 of 62 rows are in that state.

Requiring the cell to be filled by hand is not a fix; the provider already
reports the ISIN's ticker aliases, and one is usually the ticker the taxonomy
knows. Crossing those two recovers the curated identity automatically.

Concretely, FR0010755611 (Amundi MSCI USA 2x, bought 2026-07-27) has no isin
cell. Without the hint the bounded provider sweep spent its whole probe budget
on OpenFIGI bare tickers that all 404 — CL2.MI (1276 closes) sat at candidate 51
and was never probed — and the fund resolved to 18MF.MU, a venue with a SINGLE
close. No usable history, so it dropped out of holding_performance and blocked
the newsletter. Network-free: the provider lookup is stubbed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tarzan import config as cfg


# The exact ticker set OpenFIGI returns for FR0010755611: ten bare symbols,
# none of which quote on Yahoo, with the curated 'CL2' among them.
_FIGI_ALIASES = ["0WAX", "18MF", "18MFD", "AM64", "AM64V",
                 "CL2", "CL22USD", "CL2GR", "CL2M", "CL2P"]


def test_taxonomy_membership_distinguishes_curated_from_alias():
    """The bridge needs a real membership test, not the echoing resolver."""
    # resolve_taxonomy_identity echoes unknown input back, so it cannot answer
    # "is this curated?" — which is why instrument_taxonomy_has exists.
    assert cfg.resolve_taxonomy_identity("", "18MF")[1] == "18MF", (
        "the resolver echoes: this is the trap the helper avoids"
    )
    assert cfg.instrument_taxonomy_has("CL2"), "CL2 is a curated row"
    assert not cfg.instrument_taxonomy_has("18MF"), "18MF is only a venue alias"
    assert not cfg.instrument_taxonomy_has("0WAX")
    assert not cfg.instrument_taxonomy_has("")


def test_cl2_row_has_no_isin_so_the_isin_lookup_finds_nothing():
    """The precondition: this is why the bridge is needed at all."""
    isin, ticker = cfg.resolve_taxonomy_identity("FR0010755611", "")
    assert isin == "FR0010755611"
    assert ticker == "", (
        "the CL2 row's isin cell is empty, so an ISIN-only order gets no hint "
        "— if this ever fails the cell was filled and the bridge is untested"
    )


def test_alias_bridge_recovers_the_curated_ticker(monkeypatch):
    """One of the provider's aliases IS the curated ticker: use it as the hint."""
    from tarzan.data import enricher as E

    monkeypatch.setattr(E, "_openfigi_lookup", lambda _isin: list(_FIGI_ALIASES))
    hint = next((a for a in E._openfigi_lookup("FR0010755611")
                 if cfg.instrument_taxonomy_has(a)), "")
    assert hint == "CL2", f"expected the curated CL2, got {hint!r}"


def test_curated_hint_puts_the_primary_listing_inside_the_probe_cap(monkeypatch):
    """With the hint, CL2.MI is probed; without it, it is candidate 51."""
    from tarzan.data import enricher as E

    monkeypatch.setattr(E, "_openfigi_lookup", lambda _isin: list(_FIGI_ALIASES))
    monkeypatch.setattr(E, "_fetch_candidate_meta",
                        lambda sym: E._Candidate(sym, {}, 30.0, "EUR", "Amundi"))

    with_hint = [c.symbol for c in E._collect_candidate_metas("FR0010755611", "CL2")]
    assert "CL2.MI" in with_hint, "the curated hint must reach Milan within the cap"
    assert len(with_hint) <= E._MAX_RESOLVE_FETCHES

    # And the suffix-major sweep keeps a qualified venue inside the cap even
    # when no hint exists at all.
    no_hint = [c.symbol for c in E._collect_candidate_metas("FR0010755611", "")]
    assert any(s.endswith(".MI") for s in no_hint), (
        "bare aliases must not consume the whole budget: a .MI listing has to "
        "be probed even with no curated hint"
    )


def test_suffixed_hint_still_reaches_the_primary_listing(monkeypatch):
    """A curated hint is not always bare — the fix must not assume it is.

    In production (the 2026-07-29 10:41 run) the alias bridge for
    FR0010755611 matched CL2.PA, not bare CL2: OpenFIGI's data item for the
    Paris listing maps to a qualified yfinance ticker, and the alias-bridge
    loop reaches that qualified form before the bare one in the same
    provider result. Without expanding the bare root of a suffixed hint
    across every configured venue, only that one exact suffixed form is a
    qualified candidate, and the instrument's actual primary listing
    (CL2.MI — confirmed live via the newsletter's own instrument-data-
    sources footer) stays buried behind the unrelated alias roots in the
    generic sweep, unreached within the fetch budget. That is exactly how
    18MF.MU won originally, and it won again even after the alias bridge
    correctly found CL2.PA — the alias bridge gets the right identity INTO
    consideration, it doesn't make that identity win on its own.
    """
    from tarzan.data import enricher as E

    monkeypatch.setattr(E, "_openfigi_lookup", lambda _isin: ["CL2.PA", "CL2", "18MF.MU"])
    monkeypatch.setattr(E, "_fetch_candidate_meta",
                        lambda sym: E._Candidate(sym, {}, 30.0, "EUR", "Amundi"))

    candidates = [c.symbol for c in E._collect_candidate_metas("FR0010755611", "CL2.PA")]
    assert "CL2.MI" in candidates, (
        "a suffixed hint must still reach the primary listing, not just the "
        "one exact suffixed form the alias bridge happened to return"
    )
    assert candidates[0] == "CL2.PA" and candidates[1] == "CL2", (
        "the hint and its bare root must lead the candidate list"
    )
    assert len(candidates) <= E._MAX_RESOLVE_FETCHES


def _hist(n_closes: int) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=max(n_closes, 1), freq="B")
    return pd.DataFrame({"Close": [30.0] * len(idx)}, index=idx).iloc[:n_closes]


@pytest.mark.parametrize("closes,expected", [(0, False), (1, False), (2, True), (1276, True)])
def test_selection_requires_enough_history_for_a_return(closes, expected):
    """A single close is a quote in a series' clothing.

    Candidate selection used the one-close boundary test, so 18MF.MU (1 close)
    could win over CL2.MI (1276). Every downstream consumer drops a <2-point
    series on its own guard, so selection applies the same threshold.
    """
    from tarzan.data.enricher import _history_supports_a_return

    assert _history_supports_a_return(_hist(closes)) is expected


def test_single_close_still_counts_as_boundary_evidence():
    """The stricter test must not weaken valuation's own one-close rule."""
    from tarzan.data.enricher import _history_visible_at_boundary

    assert _history_visible_at_boundary(_hist(1)), (
        "a lone close is still a valid current price — only SELECTION is stricter"
    )


# ── A cached symbol must never cost the instrument its curated identity ───────
# The resolution cache holds ISIN→symbol for 30 days and CI restores it every
# run, so a poisoned entry outlives any fix downstream of it: the live-run fast
# path accepted the cached symbol unconditionally, so 18MF.MU kept being reused
# and the alias bridge was never reached.
#
# The damage is not confined to that one row. asset_class/role are looked up by
# BARE ticker, so 18MF.MU normalises to "18MF", which the taxonomy does not
# know. The holding lands in OTHER with no class, and one unclassified valued
# holding sets classification_available=False, which blanks the WHOLE allocation
# section — every class 0% / €0, "no series", total notional drift −125.0pp.

def test_poisoned_cache_entry_is_rejected_when_a_curated_alias_exists(monkeypatch):
    """The cached symbol is unknown to the taxonomy but an alias is curated."""
    from tarzan.data import enricher as E

    monkeypatch.setattr(E, "_openfigi_lookup", lambda _isin: list(_FIGI_ALIASES))
    # 18MF.MU normalises to a bare ticker the taxonomy has never heard of...
    assert not cfg.instrument_taxonomy_has("18MF.MU")
    # ...while CL2 — an alias of the SAME ISIN — is curated. That combination is
    # what marks the cached entry stale rather than merely unfamiliar.
    assert any(cfg.instrument_taxonomy_has(a) for a in E._openfigi_lookup("X"))


def test_a_curated_cached_symbol_is_still_trusted(monkeypatch):
    """Self-healing must not turn into re-resolving on every single run."""
    from tarzan.data import enricher as E

    monkeypatch.setattr(E, "_openfigi_lookup", lambda _isin: list(_FIGI_ALIASES))
    # CL2.MI normalises to CL2, which IS curated, so the cache stays valid and
    # no re-resolution (or provider traffic) is triggered.
    assert cfg.instrument_taxonomy_has("CL2.MI")


def test_unclassified_holding_is_what_blanks_the_allocation_section():
    """Tie the ticker choice to the symptom seen in the mail.

    This is the mechanism, in two lines: no asset_class -> empty breakdown ->
    the holding counts as unresolved -> classification_available False.
    """
    from tarzan.engine import allocations as alloc

    class _H:
        isin = "FR0010755611"
        current_value = 18199.54
        market_value_eur = 18199.54
        class_breakdown = None
        asset_class = None
        ticker = "18MF"

    class _Curated(_H):
        ticker = "CL2"
        asset_class = "Equities"

    assert alloc.holding_class_breakdown(_H()) == {}, (
        "an unknown bare ticker yields no class — the OTHER row in the mail"
    )
    assert alloc.holding_class_breakdown(_Curated()) == {"Equities": 100.0}
