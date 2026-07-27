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
}
