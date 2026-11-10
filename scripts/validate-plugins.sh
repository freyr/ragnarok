#!/usr/bin/env bash
# Validate the marketplace and every plugin it lists with the Claude Code CLI, check that
# the marketplace and the plugin folders agree, and check that all versions are the same.
set -euo pipefail

cd "$(dirname "$0")/.."

marketplace=.claude-plugin/marketplace.json
failed=0

fail() {
  echo "::error::$1"
  failed=1
}

echo "== claude plugin validate --strict $marketplace"
claude plugin validate --strict . || fail "marketplace validation failed"

# Every plugin the marketplace lists, as a folder path (one per line; plugin paths have no spaces).
listed=$(jq -r '.plugins[].source | select(type == "string") | sub("^\\./"; "")' "$marketplace" | sort)

# Every plugin folder in the repository.
folders=$(find plugins -mindepth 1 -maxdepth 1 -type d | sort)

for folder in $folders; do
  if ! printf '%s\n' "$listed" | grep -qx "$folder"; then
    fail "$folder is not listed in $marketplace"
  fi
done

expected_version=$(jq -r '.metadata.version // empty' "$marketplace")
if [ -z "$expected_version" ]; then
  fail "$marketplace has no metadata.version"
fi

for folder in $listed; do
  manifest="$folder/.claude-plugin/plugin.json"
  if [ ! -f "$manifest" ]; then
    fail "$folder is listed in $marketplace but has no $manifest"
    continue
  fi

  name=$(jq -r '.name' "$manifest")
  listed_name=$(jq -r --arg src "./$folder" '.plugins[] | select(.source == $src) | .name' "$marketplace")
  if [ "$name" != "$listed_name" ]; then
    fail "$manifest has name '$name', but $marketplace lists '$listed_name'"
  fi

  version=$(jq -r '.version // empty' "$manifest")
  if [ "$version" != "$expected_version" ]; then
    fail "$manifest has version '$version', expected '$expected_version' (metadata.version in $marketplace)"
  fi

  echo "== claude plugin validate --strict $folder"
  claude plugin validate --strict "$folder" || fail "$folder: strict validation failed"

  echo "== claude plugin validate --profile directory $folder"
  claude plugin validate --profile directory "$folder" || fail "$folder: plugin directory checks failed"

  if find "$folder" -name '*.test.ts' -o -name '*.test.tsx' | grep -q .; then
    echo "== claude plugin test $folder"
    claude plugin test "$folder" || fail "$folder: tests failed"
  fi
done

if [ "$failed" -ne 0 ]; then
  exit 1
fi
echo "All plugins are valid at version $expected_version."
