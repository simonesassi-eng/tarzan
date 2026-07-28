"""The one newsletter palette.

A leaf module with no Tarzan imports, so both the newsletter package and
``tarzan.export._charts`` can read the same colours. ``_charts`` previously
duplicated seven of these as module constants, because importing them from
``newsletter._constants`` pulls in the newsletter package ``__init__`` and
closes an import cycle. Duplication meant a palette change had to be made in
two places and the chart axes could silently drift from the tables around them.

Keys are named by ROLE, not by appearance, so a different palette can be
substituted without renaming anything at the call sites:

``card``     surface a table row sits on
``row_rule`` horizontal rule under a table row
``col_rule`` vertical rule between value columns, lighter than ``row_rule``
``zebra``    alternating row stripe
``head_bg``  column-header band
``group_bg`` group-header row (asset class · role)

Those three existed only as literals inline in the shared table renderer
(``#FFFFFF``, ``#F1F2F8``, ``#EFF1F8``), which is why the tables could not
follow the palette at all.
"""

from __future__ import annotations

PALETTE = {
    # Terminal palette: dark surfaces, monospace-friendly contrast. Roles are
    # unchanged from the light set, so nothing at the ~140 call sites moved.
    "accent": "#6E9BFF",
    "ink": "#E6EDF6",
    "muted": "#8FA3BC",
    "subtle": "#66798F",
    "page": "#05090D",
    "card": "#0C131B",
    "card_alt": "#111A24",
    # Table furniture, distinct from ``card_alt``. The alternating row stripe
    # used ``card_alt``, which is also the surface of every card, so a striped
    # row read as a nested panel; and the group header row used ``card``, so it
    # was invisible against the row below it.
    "zebra": "#0F1720",
    "head_bg": "#16212D",
    "group_bg": "#0F1821",
    "border": "#22303F",
    "row_rule": "#1A2531",
    "col_rule": "#16212D",
    "green": "#2FBF71",
    "amber": "#E5A038",
    "red": "#FF5F52",
    "green_bg": "#0F2A1D",
    "green_border": "#1B4332",
    # Whole-row BUY/SELL tints in the Optimizer: a shade off the card rather
    # than a wash, which on a dark surface would swamp the text.
    "green_tint": "#0E1F18",
    "red_tint": "#1E1211",
    "amber_bg": "#2A1F0C",
    "red_bg": "#2C1210",
    "accent_bg": "#132038",
    # Chart-only roles, kept distinguishable from axis furniture.
    "bench": "#8FA3BC",
    "pnl": "#38BDF8",
    # Unrealized P&L sits beside Total P&L on every return chart, so it needs a
    # hue that reads as its sibling without being confusable with it: violet
    # against the cyan of ``pnl``, and far from green/red (which mean sign, not
    # series) and from the grey of ``bench``.
    "unreal": "#A78BFA",
}
