# ragnarok

Claude Code support for [ragnarok](https://github.com/freyr/ragnarok-bundle), the local RAG search over Obsidian
vaults. The plugin runs the ragnarok MCP server, sets ragnarok up on a computer, adapts it to your vaults,
and tells Claude to search the vaults with ragnarok before it uses Grep.

## Install

```text
/plugin marketplace add freyr/ragnarok-bundle
/plugin install ragnarok@ragnarok
```

Then ask Claude to set up ragnarok. The `ragnarok-setup` skill does the rest.

## Upgrade from ragnarok-mcp 1.x

Close every Claude Code session before the first start with ragnarok-mcp 2.0.0. Each session runs
its own MCP server. The first 2.0.0 start migrates the index in place, with no new embedding; a 1.x
server that starts after it adds the removed tables back and asks for a full rebuild.

## Content

| Part | What it does |
| --- | --- |
| MCP server `ragnarok` | `uvx --from "ragnarok-mcp>=2.0.0" ragnarok-mcp serve` from [PyPI](https://pypi.org/p/ragnarok-mcp). The tools are `mcp__plugin_ragnarok_ragnarok__*`. |
| Skill `ragnarok-setup` | Runs `uvx ragnarok-mcp init` (Ollama service, embedding model, index) and `ragnarok doctor`, fixes what fails, and moves an older checkout setup to the plugin. |
| Skill `ragnarok-adapt` | Adapts ragnarok to your vaults: survey, private folders, path rules, optional frontmatter backfill, an evaluation set. |
| SessionStart hook | Adds the "ragnarok first" search guidance to each session, so Claude uses the ragnarok tools before Grep. |
| `scripts/ragnarok-home.sh` | Prints the home folder of the server: `$RAGNAROK_HOME`, else `~/.ragnarok`. |

## Home folder

The home folder holds `config.toml`, the index and the server log. It is `~/.ragnarok`. To use a different
folder, for example an older checkout of the ragnarok server, set `RAGNAROK_HOME` in the `env` of
`~/.claude/settings.json` and in your shell profile.

## Requirements

macOS with Homebrew, `brew install ollama uv`, and Claude Code.

## Note

If your `~/.claude/CLAUDE.md` already holds a "ragnarok first" block, remove it. The hook gives the same
guidance, and two copies that differ confuse Claude. If `ragnarok init` registered the server at the user
scope before, remove that registration (`claude mcp remove --scope user ragnarok`, or `uvx ragnarok-mcp init`
asks to remove it), so that only the plugin server runs.
