"""One boundary-aware identity catalog shared by target joins and seed checks."""

from dataclasses import dataclass

from tarzan import config, runtime
from tarzan.data import price_cache
from tarzan.models.instrument_key import instrument_key, normalize_isin, normalize_ticker


def identity_aliases(isin, ticker) -> set[str]:
    return {
        instrument_key(isin, ticker), normalize_isin(isin),
        normalize_ticker(ticker), str(ticker or "").strip().upper(),
    } - {""}


@dataclass(frozen=True)
class HoldingIdentity:
    isin: str
    tickers: frozenset[str]
    aliases: frozenset[str]


class TargetIdentityCatalog:
    """No transport or price/name guesses; one read per identity in a phase."""

    def __init__(self):
        self.boundary = runtime.as_of() or runtime.context().captured_at
        self._profiles = {}
        self._rows = {}
        self._holdings = {}

    def resolve_row(self, row):
        key = id(row)
        if key not in self._rows:
            self._rows[key] = config.resolve_taxonomy_identity(
                row.get("isin"), row.get("ticker"), row.get("name"),
            )
        return self._rows[key]

    def profile(self, isin):
        if isin not in self._profiles:
            self._profiles[isin] = price_cache.load_instrument_profile(
                isin, as_of=self.boundary, allow_stale=True,
            )
        return self._profiles[isin]

    def holding(self, holding):
        key = (holding.isin, holding.ticker, holding.name)
        if key in self._holdings:
            return self._holdings[key]
        isin = normalize_isin(holding.isin)
        tickers = set()
        aliases = identity_aliases(holding.isin, holding.ticker)
        if holding.ticker and str(holding.ticker).strip().upper() != isin:
            tickers.add(str(holding.ticker).strip().upper())
        if isin:
            reverse = price_cache.load_ticker_isin_reverse(isin)
            if reverse:
                tickers.add(str(reverse).strip().upper())
            profile = self.profile(isin)
            if profile:
                tickers.update(str(t).strip().upper() for t in profile.get("tickers", ()) if t)
        elif holding.ticker:
            isin = normalize_isin(price_cache.load_ticker_isin(holding.ticker) or "")
        curated_isin, curated_ticker = config.resolve_taxonomy_identity(
            holding.isin, holding.ticker, holding.name,
        )
        isin = isin or normalize_isin(curated_isin)
        if curated_ticker:
            tickers.add(str(curated_ticker).strip().upper())
        for ticker in tickers:
            aliases.update(identity_aliases(isin, ticker))
        aliases.update(identity_aliases(curated_isin, curated_ticker))
        result = HoldingIdentity(isin, frozenset(tickers), frozenset(aliases))
        self._holdings[key] = result
        return result
