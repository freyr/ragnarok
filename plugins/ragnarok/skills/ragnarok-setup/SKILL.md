---
name: ragnarok-setup
description: Use to install ragnarok (local RAG search over Obsidian vaults) on a computer, to repair it, or to check it — when the mcp__plugin_ragnarok_ragnarok__* tools are absent or fail, when `ragnarok doctor` reports a failure, or when the person asks to set up vault search on a new Mac. Runs `uvx ragnarok-mcp init` and `doctor`, and fixes what fails. Not for tuning the path rules or private folders (ragnarok-adapt) or for normal searching.
---

# Set up ragnarok

ragnarok is a local MCP server that searches the Markdown notes of Obsidian vaults. It is the PyPI package
`ragnarok-mcp` ([freyr/ragnarok-bundle](https://github.com/freyr/ragnarok-bundle)). This plugin starts the server with
`uvx --from "ragnarok-mcp>=2.0.0" ragnarok-mcp serve`, so the tools are `mcp__plugin_ragnarok_ragnarok__*`. It also gives the search
guidance (a SessionStart hook) and these skills.

## The home folder

The home folder holds `config.toml`, `data/index.db` and `logs/`. It is `$RAGNAROK_HOME`, else
`~/.ragnarok`. The server and each `uvx ragnarok-mcp` command use the same folder only when they see the same
`RAGNAROK_HOME`. For a folder other than `~/.ragnarok`, set it in the `env` of `~/.claude/settings.json`
(the server and Claude's shell read it there) and in the shell profile of the person.

```bash
H=$(bash "${CLAUDE_PLUGIN_ROOT}/scripts/ragnarok-home.sh") && echo "home: $H" || echo "no config.toml in $H"
```

## Check

```bash
uvx ragnarok-mcp doctor
```

`doctor` checks the configuration, Ollama, the Ollama service, the embedding model, the index and the
MCP server. It prints a fix for each failure. When all parts pass and the tools are still absent, the
person must restart Claude Code.

## Install on a new computer

Requirements: macOS with Homebrew, `brew install ollama uv`.

1. Ask the person for the vault root (the folder that holds the vault folders, for example
   `~/Documents/Vaults`), and whether the home folder `~/.ragnarok` is good.
2. Tell the person what `init` does before you run it: it writes `config.toml` with each folder under the
   root as a vault, starts Ollama with `brew services`, pulls the embedding model and indexes all vaults.
   The plugin gives the MCP server, so init registers nothing. `--yes` accepts the default answer of each
   question, so ask the person first. Without `--yes`, the person runs it in a terminal
   (`! uvx ragnarok-mcp init`).
3. Run the setup. The first index of 1,500 notes takes about 12 minutes, so run it in the background:

   ```bash
   uvx ragnarok-mcp init --yes --root "<vault root>"
   ```

4. Run `uvx ragnarok-mcp doctor` until all parts pass.
5. Before the first search, agree on the private folders with the person (`ragnarok-adapt`, step 2).
   Private patterns apply when the server answers, so notes indexed before the patterns are safe.
6. Tell the person to restart Claude Code. The tools then appear as `mcp__plugin_ragnarok_ragnarok__*`.
7. Continue with the `ragnarok-adapt` skill to set the path rules for the vaults of the person.

## Move from a checkout

An older setup has a checkout of the ragnarok server as its home folder and a user registration
`uv --directory <checkout> run ragnarok serve`. To keep its configuration and index:

1. Set `RAGNAROK_HOME` to the checkout folder in the `env` of `~/.claude/settings.json` and in the shell
   profile.
2. Run `uvx ragnarok-mcp init`. Step 6 sees the plugin server and asks to remove the user registration
   `ragnarok`. The tools `mcp__ragnarok__*` then go away.
3. Remove a "ragnarok first" block from `~/.claude/CLAUDE.md` that names `mcp__ragnarok__*`.
4. Restart Claude Code.

## Repair

| Symptom | Fix |
| --- | --- |
| `doctor`: Ollama not running | `brew services start ollama` |
| `doctor`: model missing | `ollama pull` the model that `config.toml` names |
| `doctor`: two servers, the plugin and a registration | `claude mcp remove --scope user ragnarok` |
| `doctor`: no config or no index, but the server has data | The shell and the server see different `RAGNAROK_HOME` values. Set it in both places (see The home folder). |
| `uvx` not found | `brew install uv`, then restart Claude Code |
| Tools present, results empty or old | `uvx ragnarok-mcp status`, then `uvx ragnarok-mcp index` |
| A slow first call after a config change | Normal: the call indexes the changed notes. Run `uvx ragnarok-mcp index` first. |

The server log is `$H/logs/server.log`, one line for each tool call.

## Update

```bash
uvx ragnarok-mcp@latest doctor
```

This puts the newest release in the uv cache, and the server uses it at the next start. Restart Claude
Code after an update.
