#!/usr/bin/env bash
# Print the ragnarok home folder that the plugin's MCP server uses: $RAGNAROK_HOME, else ~/.ragnarok.
# The folder holds config.toml, data/ and logs/. Exit 1 when it has no config.toml (it still prints the path).
set -uo pipefail

home=${RAGNAROK_HOME:-$HOME/.ragnarok}
home=${home/#\~/$HOME}
printf '%s\n' "$home"
[ -f "$home/config.toml" ]
