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
    "accent": "#5B5BD6",
    "ink": "#1E293B",
    "muted": "#64748B",
    "subtle": "#94A3B8",
    "page": "#F1F2F8",
    "card": "#FFFFFF",
    "card_alt": "#F8FAFF",
    "border": "#E5E7EF",
    "row_rule": "#F1F2F8",
    "col_rule": "#EFF1F8",
    "green": "#15803D",
    "amber": "#D97706",
    "red": "#DC2626",
    "green_bg": "#DCFCE7",
    "green_border": "#BBF7D0",
    # Very light action tints for whole-row BUY/SELL backgrounds in the
    # Optimizer (green-50 / red-50) — softer than the *_bg pills above.
    "green_tint": "#ECFDF5",
    "red_tint": "#FEF2F2",
    "amber_bg": "#FFF7ED",
    "red_bg": "#FEE2E2",
    "accent_bg": "#EEF2FF",
    # Chart-only roles. The benchmark line and the P&L line are deliberately
    # not `muted`/`accent`: they must stay distinguishable from axis furniture.
    "bench": "#94A3B8",
    "pnl": "#0EA5E9",
}
