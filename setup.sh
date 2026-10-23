#!/usr/bin/env bash
# Set up ragnarok on this computer from this folder: install uv if it is missing, then run `ragnarok init`.
# Arguments go to init, for example: ./setup.sh --root ~/Documents/Vaults
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null; then
  if command -v brew >/dev/null; then
    brew install uv
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

uv sync
uv run ragnarok init "$@"
