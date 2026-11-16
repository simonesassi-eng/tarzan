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
    # The Securities Statement (Nov 2026): one dark ground, no card surfaces.
    # Every "surface" role resolves to the page so tables sit on the ground like
    # ledger rows; ``head_bg``/``group_bg`` are the one shade used for header
    # bands. Signal colours are spent on signs, gaps and the market stamp only.
    "accent": "#EEF2F7",
    "blue": "#6E9BFF",
    "ink": "#EEF2F7",
    "muted": "#93A0B4",
    "subtle": "#7D8CA0",
    "faint": "#6E7D95",   # 4.6:1 on the ground: the floor for 9.5px labels
    "page": "#0B1017",
    "card": "#0B1017",
    "card_alt": "#0B1017",
    "zebra": "#0B1017",
    "head_bg": "#131C27",
    "group_bg": "#131C27",
    "shade": "#131C27",
    "border": "#2A3442",
    "row_rule": "#1B2330",
    "col_rule": "#1B2330",
    "green": "#3DBF7F",
    "amber": "#E5A038",
    "red": "#E4573D",
    "green_bg": "#0F2A1D",
    "green_border": "#1B4332",
    "green_tint": "#0E1F18",
    "red_tint": "#1E1211",
    "amber_bg": "#2A1F0C",
    "red_bg": "#2C1210",
    "accent_bg": "#15202E",
    "bench": "#8A98AB",
    "pnl": "#38C8FF",
    "unreal": "#A855F7",
    "port": "#EEF2F7",
    "target": "#F472B6",
}

# Asset-class hues, from the one source every surface reads
# (``tarzan.export._format.ASSET_CLASS_COLORS``), as CSS colours. Used
# identically everywhere a class is named: the group band (at 20%), the 2px
# rail beside its rows, the dot before its name and the bridge slices.
from tarzan.export._format import ASSET_CLASS_COLORS as _ASSET_CLASS_COLORS

CLASS_COLOURS = {k: f"#{v}" for k, v in _ASSET_CLASS_COLORS.items()}


def mix(base: str, target: str, t: float) -> str:
    """Linear mix of two #RRGGBB colours, ``t`` of the way from base to target."""
    def ch(s, i):
        return int(s[1 + 2 * i:3 + 2 * i], 16)
    return "#{:02X}{:02X}{:02X}".format(*(round(ch(base, i) + t * (ch(target, i) - ch(base, i)))
                                          for i in range(3)))


def class_band(cls: str, t: float = 0.20) -> str:
    """The faint band behind a class header: the class hue over the ground."""
    c = CLASS_COLOURS.get(cls)
    return mix(PALETTE["page"], c, t) if c else PALETTE["shade"]

# The one type stack. Declared at EVERY text-bearing element, not inherited from
# ``<body>``: the clients that drop body/``<style>`` font rules (Outlook's Word
# engine, some webmail) otherwise render the elements that restate it in
# monospace and everything else in Times, splitting one document across two
# typefaces. Three inconsistent mono stacks and a sans-serif one used to be
# inline in the charts and tables; a stack change had to be made in four places.
FONT_STACK = '-apple-system,BlinkMacSystemFont,"Helvetica Neue",Helvetica,Arial,sans-serif'
# Codes only: ISINs, feed tickers in the appendix. Never running text or figures.
MONO_STACK = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"

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
    "display": "font-size:34px;font-weight:700;line-height:1.05;letter-spacing:-0.01em;",
    "figure": "font-size:19px;font-weight:700;line-height:1.1;",
    "title": "font-size:12.5px;font-weight:700;letter-spacing:0.12em;"
             "text-transform:uppercase;",
    "prose": "font-size:12px;font-weight:400;line-height:1.45;",
    "note": "font-size:10.5px;font-weight:400;line-height:1.4;",
    "data": "font-size:11px;font-weight:600;line-height:1.25;",
    "label": "font-size:10px;font-weight:700;letter-spacing:0.08em;"
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
    "display": 34,
    "figure": 19,
    "title": 12.5,
    "prose": 12,
    "note": 10.5,
    "data": 11,
    "label": 10,
}
