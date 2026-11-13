# What "a figure from the real book" looks like. Sourced by commit-msg and
# pre-commit beside _money.sh.
#
# _money.sh only knows euro amounts, so allocation weights ("35% NTSG") and the
# book's own returns ("YTD read +23%") went out in commit messages. This pattern
# is built at commit time from the git-ignored input/ files, so the hook knows
# the real figures without the repository ever containing them:
#
#   * every target weight next to its ticker, in either order;
#   * a return figure right after YTD / TWR / MWR / P&L.
#
# Mark a deliberate stand-in with "synthetic" on the line (pre-commit), or
# escape a whole commit with git commit --no-verify.

_root=$(git rev-parse --show-toplevel 2>/dev/null)
_targets="$_root/input/targets_per_holding.csv"

BOOK_PATTERN='(YTD|TWR|MWR|P&L)[^0-9]{0,12}[0-9]+([.,][0-9]+)?%'
if [ -f "$_targets" ]; then
	_weights=$(awk -F, 'NR > 1 && $3 != "" && $6 + 0 > 0 {
		w = $6; gsub(/\./, "[.,]", w)
		printf "|%s[^0-9]{0,20}%s ?%%|(^|[^0-9.,])%s ?%%[^0-9]{0,20}%s", $3, w, w, $3
	}' "$_targets")
	BOOK_PATTERN="$BOOK_PATTERN$_weights"
fi
