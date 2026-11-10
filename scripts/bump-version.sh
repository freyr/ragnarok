#!/usr/bin/env bash
# The jq filters below use the jq variable $v, which must not expand in the shell.
# shellcheck disable=SC2016
# Set one version on the marketplace and on every plugin it lists.
# Usage: scripts/bump-version.sh 1.2.3
set -euo pipefail

cd "$(dirname "$0")/.."

version=${1:?usage: bump-version.sh <version>}
if ! [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "::error::'$version' is not a semantic version (MAJOR.MINOR.PATCH)" >&2
  exit 1
fi

marketplace=.claude-plugin/marketplace.json

set_version() {
  local file=$1 filter=$2 tmp
  tmp=$(mktemp)
  jq --indent 2 --arg v "$version" "$filter" "$file" >"$tmp"
  mv "$tmp" "$file"
  echo "$file -> $version"
}

set_version "$marketplace" '.metadata.version = $v'

jq -r '.plugins[].source | select(type == "string")' "$marketplace" | while read -r source; do
  set_version "${source#./}/.claude-plugin/plugin.json" '.version = $v'
done
