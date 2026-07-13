#!/usr/bin/env bash
# Generate a newsletter (with the AI summary) for the current portfolio, on
# demand. Run it whenever you want a fresh digest:
#
#     bash scripts/newsletter.sh          # or: ./scripts/newsletter.sh
#
# Writes output/<YYYY-MM-DD>/portfolio_digest_<HHMM>.html and opens it (macOS).
# Loads the Gemini key from .private/gemini_key so the "Market context" and
# "Why you're diverging" AI blocks are populated; without a key it still
# renders, just without those blocks.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ORDERS="input/order_list.csv"
CONFIG="input/targets.csv"
TPH="input/targets_per_holding.csv"

if [ ! -f "$ORDERS" ]; then
  echo "No $ORDERS found. Put your order list there first." >&2
  exit 1
fi

# Load the Gemini key (enables the AI summary) if not already in the env.
if [ -z "${GEMINI_API_KEY:-}" ] && [ -f .private/gemini_key ]; then
  GEMINI_API_KEY="$(tr -d '[:space:]' < .private/gemini_key)"
  export GEMINI_API_KEY
fi
[ -n "${GEMINI_API_KEY:-}" ] && echo "AI summary: ON" || echo "AI summary: OFF (no .private/gemini_key)"

PY="$(command -v python3 || command -v python)"
CFG_ARG=(); [ -f "$CONFIG" ] && CFG_ARG=(--input_config "$CONFIG")
TPH_ARG=(); [ -f "$TPH" ] && TPH_ARG=(--input_targets_per_holding "$TPH")

# Foreground so you see progress and know when it's done. Capture the CLI's
# "Newsletter saved to: <path>" line to open the result.
LOG="$(mktemp)"
"$PY" -m tarzan.main --input_orders "$ORDERS" "${CFG_ARG[@]}" "${TPH_ARG[@]}" 2>&1 | tee "$LOG"

OUT="$(grep -oE 'Newsletter saved to: .*\.html' "$LOG" | tail -1 | sed 's/^Newsletter saved to: //')"
rm -f "$LOG"
if [ -n "$OUT" ] && [ -f "$OUT" ]; then
  echo ""
  echo "Newsletter: $OUT"
  command -v open >/dev/null 2>&1 && open "$OUT"   # macOS: open in the browser
fi
