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

# The one type stack. Declared at EVERY text-bearing element, not inherited from
# ``<body>``: the clients that drop body/``<style>`` font rules (Outlook's Word
# engine, some webmail) otherwise render the elements that restate it in
# monospace and everything else in Times, splitting one document across two
# typefaces. Three inconsistent mono stacks and a sans-serif one used to be
# inline in the charts and tables; a stack change had to be made in four places.
FONT_STACK = "ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace"

# Five sizes (22/13/11/10/9), six roles, named by ROLE like the palette
# keys above.
#
# The digest reached fourteen CSS sizes and six SVG sizes because every module
# invented its own value for the same logical role: a table value existed at
# 9.5px (returns), 10px (markets) and 10.5px (holdings); an uppercase
# micro-label at 9px/0.09em, 9.5px/0.06em, 9px/0.08em and 7.5px/none. Half-pixel
# steps are below the perceptual threshold at these sizes, so they read as ragged
# rather than layered, and 516 elements sat at 8.5px or below where mid-grey on a
# near-black surface stops being legible on a phone.
#
# Each value is a complete declaration block, so a call site sets the role and
# then only what is genuinely local to it (colour, alignment, margin):
#
#     f'<div style="{TYPE["label"]}color:{PALETTE["muted"]};">NET FLOW</div>'
#
# ``figure`` and ``title`` share a size deliberately: they are one tier,
# separated by weight and case rather than by a fifth of a pixel. Weight is the
# axis the digest never used — everything was 600 or 700, so SIZE had to carry
# every level of hierarchy on its own, which is how fourteen of them accumulated.
# Running text at 400 against data at 600 and labels at 700 does that job with
# one size each.
TYPE = {
    "display": "font-size:22px;font-weight:700;line-height:1.15;",
    "figure": "font-size:13px;font-weight:700;line-height:1.25;",
    "title": "font-size:13px;font-weight:700;letter-spacing:0.06em;"
             "text-transform:uppercase;",
    # ALL running text outside the tables: the narrative, section subtitles,
    # callouts, methodology sentences, captions, legends, disclaimers, the
    # footer. There is deliberately no second prose size -- a subtitle and the
    # caption three lines below it are the same kind of sentence, and setting
    # one 2px larger than the other read as two typefaces sharing a section.
    # What separates them is COLOUR: ink for a callout, muted for a subtitle,
    # subtle for fine print.
    "prose": "font-size:11px;font-weight:400;line-height:1.5;",
    "data": "font-size:10px;font-weight:600;line-height:1.25;",
    "label": "font-size:9px;font-weight:700;letter-spacing:0.06em;"
             "text-transform:uppercase;",
}

# The same scale as bare numbers, for SVG ``font-size`` attributes (no unit) and
# for the few call sites that compose a size into a larger declaration. Charts
# are drawn at their rendered width (viewBox 580/282 in a 580px column), so one
# user unit is one CSS pixel there and the two forms agree. Sparklines are NOT:
# they are 66-unit viewBoxes stretched to the cell with
# ``preserveAspectRatio="none"``, so a font-size in one is neither px nor square
# — which is why no sparkline draws text.
TYPE_PX = {
    "display": 22,
    "figure": 13,
    "title": 13,
    "prose": 11,
    "data": 10,
    "label": 9,
}
