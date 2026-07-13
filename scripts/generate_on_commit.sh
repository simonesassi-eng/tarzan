#!/usr/bin/env bash
# Generate a newsletter (with the AI summary) for the current portfolio.
#
# Invoked by the git post-commit hook so every new commit leaves a fresh
# output/<YYYY-MM-DD>/portfolio_digest_*.html you can open. Also runnable by
# hand:  bash scripts/generate_on_commit.sh
#
# Best-effort by design: it must never block or fail a commit. It loads the
# Gemini key from .private/gemini_key (so the "Market context" + "Why you're
# diverging" AI blocks are populated), runs the CLI, and logs to
# output/generate_on_commit.log. If inputs are missing it skips quietly.
set -u

# Repo root = parent of this script's dir, regardless of where git invokes it.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 0

ORDERS="input/order_list.csv"
CONFIG="input/targets.csv"
TPH="input/targets_per_holding.csv"
LOG="output/generate_on_commit.log"
mkdir -p output

# No order list → nothing to report. Skip silently (e.g. a fresh clone / CI).
if [ ! -f "$ORDERS" ]; then
  echo "[skip] $ORDERS not found — no newsletter generated." >> "$LOG"
  exit 0
fi

# Load the Gemini key (enables the AI summary) if present. Absent → the
# newsletter still renders, just without the AI blocks.
if [ -z "${GEMINI_API_KEY:-}" ] && [ -f .private/gemini_key ]; then
  GEMINI_API_KEY="$(tr -d '[:space:]' < .private/gemini_key)"
  export GEMINI_API_KEY
fi

PY="$(command -v python3 || command -v python)"
[ -z "$PY" ] && { echo "[skip] no python on PATH." >> "$LOG"; exit 0; }

CFG_ARG=(); [ -f "$CONFIG" ] && CFG_ARG=(--input_config "$CONFIG")
TPH_ARG=(); [ -f "$TPH" ] && TPH_ARG=(--input_targets_per_holding "$TPH")

echo "=== $(date '+%Y-%m-%d %H:%M:%S') generating newsletter for $(git rev-parse --short HEAD 2>/dev/null) ===" >> "$LOG"

# Run in the background so a commit is never blocked by the ~1-2 min live
# price + AI fetch. Detach fully (nohup + disown) and append to the log.
nohup "$PY" -m tarzan.main \
  --input_orders "$ORDERS" "${CFG_ARG[@]}" "${TPH_ARG[@]}" \
  >> "$LOG" 2>&1 &
disown 2>/dev/null || true

echo "[started] newsletter generation running in background — see $LOG" >> "$LOG"
exit 0
