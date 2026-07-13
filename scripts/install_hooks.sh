#!/usr/bin/env bash
# Install Tarzan's git hooks into .git/hooks (which git does not track).
# Run once per clone:  bash scripts/install_hooks.sh
set -eu

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOK="$ROOT/.git/hooks/post-commit"

cat > "$HOOK" <<'EOF'
#!/usr/bin/env bash
# Auto-generate a newsletter after every commit. Best-effort, never blocks.
exec bash "$(git rev-parse --show-toplevel)/scripts/generate_on_commit.sh"
EOF
chmod +x "$HOOK"
echo "Installed post-commit hook → $HOOK"
echo "Every commit now regenerates output/<date>/portfolio_digest_*.html (with AI)."
