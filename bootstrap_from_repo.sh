#!/bin/zsh
set -euo pipefail

REPO_URL="${REPO_URL:-}"
REPO_DIR="${REPO_DIR:-$HOME/darkbloom-fleet}"
MODE="${1:-earn}"

if [[ -z "$REPO_URL" ]]; then
  echo "Usage: REPO_URL=https://github.com/<owner>/<repo>.git $0 [earn|stt]" >&2
  exit 1
fi

if [[ -d "$REPO_DIR/.git" ]]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$REPO_DIR"
fi

chmod +x "$REPO_DIR/darkbloom_fleet_setup.sh"
exec "$REPO_DIR/darkbloom_fleet_setup.sh" "$MODE"
