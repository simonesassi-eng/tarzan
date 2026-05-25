"""Convert a Fineco "Portafoglio di sintesi" export into a Tarzan
holdings CSV.

This step is **optional**. The main pipeline (`python -m tarzan.main`)
only ever reads ``input/holdings.csv``; how that file gets there is up
to the user. Two common workflows:

  1. Hand-curated: edit ``input/holdings.csv`` directly and run the
     pipeline.
  2. Automated: drop the Fineco export and the per-holding targets
     into ``input/fineco_raw/`` and run this preprocessor — it
     produces a fresh ``input/holdings.csv`` (with a timestamped
     backup of the previous one).

Default paths follow the second workflow:

Usage:
    python scripts/preprocess_fineco.py
    python scripts/preprocess_fineco.py \
        --input input/fineco_raw/portafoglio-export.xls \
        --targets input/fineco_raw/targets_per_holding.csv \
        --output input/holdings.csv

The Fineco export ships with embedded headers, totals, and a few
metadata rows we have to skip; this script reads the first sheet,
locates the data block, normalises column names to the Tarzan
schema, and merges per-holding targets from a second CSV (joined by
ISIN — the stablest key Fineco exposes).

Holdings that appear in the targets CSV but not in the Fineco
report are kept as zero-balance placeholders so the rebalancer can
still propose buys into them.

A timestamped backup of the destination file is always written
before the new file is created, so a slip can be recovered without
fishing through git.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("preprocess_fineco")


# Output column order — matches what `tarzan.data.loader` expects.
HOLDINGS_COLUMNS = [
    "name", "isin", "ticker", "currency", "quantity",
    "cost_basis_eur", "market_value_eur",
    "target_equities", "target_fixed_income", "no_buy_no_sell",
]

# Italian column names from the Fineco "Portafoglio sintesi" sheet,
# mapped to the Tarzan holdings schema. The Italian header sits on
# row 2 of the export (zero-indexed); cells below carry the values.
FINECO_COLUMN_MAP = {
    "Titolo": "name",
    "ISIN": "isin",
    "Simbolo": "ticker",
    "Valuta": "currency",
    "Quantità": "quantity",
    "Valore di carico": "cost_basis_eur",
    "Valore di mercato €": "market_value_eur",
}


def _read_fineco(path: Path) -> pd.DataFrame:
    """Read the Fineco export and return a normalised holdings frame.

    The export is brittle: row 0 is a banner, row 1 is empty, row 2
    holds the column names, and the table tail contains a "Totale"
    row plus some currency totals we must drop. We anchor parsing to
    the row where the ISIN column starts looking like ISINs, which
    is robust to small layout changes upstream.
    """
    raw = pd.read_excel(path, sheet_name=0, header=None)

    # Find the header row by looking for the cell that contains
    # "Titolo" (Italian for "Security"). That's where the data table
    # begins. Anything above is a banner, anything below is data.
    header_row = None
    for i, row in raw.iterrows():
        if any(str(cell).strip() == "Titolo" for cell in row):
            header_row = i
            break
    if header_row is None:
        raise RuntimeError(
            "Could not find the 'Titolo' header in the Fineco export — "
            "did the file format change?"
        )

    headers = [str(c).strip() for c in raw.iloc[header_row].tolist()]
    data = raw.iloc[header_row + 1:].copy()
    data.columns = headers

    # Keep only the columns we recognise, renamed to the Tarzan schema.
    keep_cols = [c for c in FINECO_COLUMN_MAP if c in data.columns]
    if "ISIN" not in keep_cols:
        raise RuntimeError("Fineco export is missing the ISIN column.")
    data = data[keep_cols].rename(columns=FINECO_COLUMN_MAP)

    # Drop rows that are not holdings: footer ("Totale", currency
    # totals, blank separators). All real rows have a 12-character
    # alphanumeric ISIN; everything else gets filtered out.
    data["isin"] = data["isin"].astype(str).str.strip()
    is_isin = data["isin"].str.match(r"^[A-Z0-9]{12}$", na=False)
    dropped = (~is_isin).sum()
    if dropped:
        logger.info("Skipped %d non-holding row(s) in the Fineco export.", dropped)
    data = data[is_isin].reset_index(drop=True)

    # Light cleanup so downstream consumers (and the eyeballed CSV)
    # see the values they expect.
    for col in ("quantity", "cost_basis_eur", "market_value_eur"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["currency"] = data["currency"].astype(str).str.strip().str.upper()
    data["name"] = data["name"].astype(str).str.strip()
    data["ticker"] = data["ticker"].astype(str).str.strip()

    return data


def _read_targets(path: Optional[Path]) -> pd.DataFrame:
    """Read the per-holding targets CSV.

    The file is keyed by ISIN; only ISIN, target_equities,
    target_fixed_income, and no_buy_no_sell are joined back into the
    holdings frame. Returns an empty frame when ``path`` is missing
    so the caller can still produce a valid (target-less) output.
    """
    if path is None or not path.exists():
        if path is not None:
            logger.warning("Targets file not found at %s — emitting holdings without targets.", path)
        return pd.DataFrame(columns=[
            "isin", "target_equities", "target_fixed_income", "no_buy_no_sell",
        ])

    df = pd.read_csv(path, dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    required = {"isin"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Targets CSV is missing required column(s): {missing}")
    df["isin"] = df["isin"].astype(str).str.strip()
    df = df[df["isin"] != ""].copy()
    return df


def _backup(path: Path, fineco_dir: Path) -> Optional[Path]:
    """Make a timestamped backup of ``path`` if it exists.

    Backups land alongside the Fineco staging files (``fineco_dir``)
    rather than next to ``input/holdings.csv``, so the personal
    snapshots stay grouped with the rest of the export artefacts and
    don't clutter the top of the input directory.
    """
    if not path.exists():
        return None
    fineco_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = fineco_dir / f"{path.stem}_{stamp}.bak{path.suffix}"
    shutil.copy2(path, backup)
    return backup


def build_holdings(
    fineco_path: Path,
    targets_path: Optional[Path],
) -> pd.DataFrame:
    """Build the merged holdings frame ready to be written as CSV."""
    fineco = _read_fineco(fineco_path)
    targets = _read_targets(targets_path)

    # Outer merge so positions present only in the targets file
    # survive as zero-balance placeholders. The rebalancer treats
    # those as buyable positions.
    merged = fineco.merge(targets, on="isin", how="outer", suffixes=("", "_target"))

    # When the holding came from the targets file only, the Fineco
    # columns are NaN. Fill them with sensible placeholder values
    # and prefer the targets-file label/ticker if the Fineco ones
    # are missing. The loader knows how to interpret a quantity-zero
    # row carrying a positive target.
    fineco_only = merged["name"].notna() & merged["quantity"].notna()
    targets_only = ~fineco_only

    placeholders_added = int(targets_only.sum())
    if placeholders_added:
        logger.info(
            "Adding %d placeholder holding(s) from the targets file "
            "(positions you want to start building).",
            placeholders_added,
        )
        merged.loc[targets_only, "quantity"] = 0
        merged.loc[targets_only, "cost_basis_eur"] = 0
        merged.loc[targets_only, "market_value_eur"] = 0
        # Prefer name/ticker from the targets file when Fineco does
        # not carry the holding yet.
        for col in ("name", "ticker"):
            target_col = f"{col}_target"
            if target_col in merged.columns:
                merged.loc[targets_only, col] = merged.loc[targets_only, target_col]
        # Currency is unknown for placeholder positions; leave it
        # blank so the loader uses its default ("EUR").
        merged.loc[targets_only, "currency"] = ""

    # Ensure target columns exist even if the targets file was empty.
    for col in ("target_equities", "target_fixed_income", "no_buy_no_sell"):
        if col not in merged.columns:
            merged[col] = ""

    # Normalise the boolean and fill blanks. The loader already
    # tolerates empty strings here.
    merged["no_buy_no_sell"] = (
        merged["no_buy_no_sell"].fillna("").astype(str).str.strip()
    )

    # Final cleanup + column order.
    out = pd.DataFrame({col: merged[col] for col in HOLDINGS_COLUMNS})
    out = out.fillna("")
    return out


def _write_holdings(df: pd.DataFrame, output: Path) -> None:
    """Write the holdings frame as CSV.

    Quotes only fields that contain a comma (the Fineco "Titolo"
    column commonly does — e.g. "BTP-30OT31 4,00"), keeping the
    output diff-friendly when the portfolio shape is unchanged.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(HOLDINGS_COLUMNS)
        for _, row in df.iterrows():
            writer.writerow([row[col] for col in HOLDINGS_COLUMNS])


def _summary(df: pd.DataFrame, fineco_path: Path, targets_path: Optional[Path]) -> str:
    n = len(df)
    with_target = int(
        (df["target_equities"].astype(str).str.strip() != "").sum()
        + (df["target_fixed_income"].astype(str).str.strip() != "").sum()
    )
    placeholders = int(
        (pd.to_numeric(df["quantity"], errors="coerce").fillna(0) == 0).sum()
    )
    parts = [
        f"{n} holding(s) merged from {fineco_path.name}",
        f"{with_target} target(s) attached"
        + (f" from {targets_path.name}" if targets_path else ""),
        f"{placeholders} zero-balance placeholder(s)",
    ]
    return " · ".join(parts)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="input/fineco_raw/portafoglio-export.xls",
        help="Path to the Fineco 'Portafoglio sintesi' .xls export.",
    )
    parser.add_argument(
        "--targets", default="input/fineco_raw/targets_per_holding.csv",
        help="Path to the per-holding targets CSV (joined by ISIN).",
    )
    parser.add_argument(
        "--output", default="input/holdings.csv",
        help="Where to write the resulting holdings CSV.",
    )
    args = parser.parse_args(argv)

    fineco_path = Path(args.input)
    targets_path = Path(args.targets)
    output_path = Path(args.output)

    if not fineco_path.exists():
        logger.error("Fineco export not found at %s", fineco_path)
        return 1

    backup = _backup(output_path, fineco_path.parent)
    if backup:
        logger.info("Backed up existing holdings to %s", backup)

    df = build_holdings(fineco_path, targets_path)
    _write_holdings(df, output_path)

    logger.info("Wrote %s", output_path)
    logger.info(_summary(df, fineco_path, targets_path if targets_path.exists() else None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
