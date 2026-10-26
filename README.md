# ragnarok

Local RAG search over Obsidian vaults for Claude Code.

ragnarok indexes the Markdown notes of one or more Obsidian vaults on your computer. A local MCP server gives
Claude Code search tools over the index. No note leaves the computer: the embeddings come from a local Ollama
model, and the index is one SQLite file.

## Features

- **Hybrid search.** Each query runs a dense search (embeddings in `sqlite-vec`) and a keyword search (SQLite
  FTS5 with BM25). Reciprocal rank fusion joins the two lists. The unit of rank is the note, with its best
  section.
- **Heading chunks.** The splitter cuts each note at its headings and joins very short sections.
- **Metadata from frontmatter or path.** Each note gets a vault, a project, an area, a type, a status and
  ticket keys. The frontmatter gives these values first. For a note without them, path rules in the
  configuration give them.
- **Project search.** `find_projects` finds the earlier projects that a prompt continues or resembles, also
  when the prompt does not name the project. `list_project` gives all notes of a project.
- **Always current.** Each tool call first scans the vaults for changed notes (usually less than 100 ms) and
  embeds them. There is no watcher process.
- **Private notes.** Notes under the configured private paths are in the index, but the server does not
  return them. Only the user can open them, with `allow_private = true` in the configuration.
- **Credential redaction.** The indexer and `open_note` replace literal passwords, tokens and keys with
  `[REDACTED]`. The vault files do not change.
- **Frontmatter backfill.** `ragnarok backfill propose` writes a proposal of the missing frontmatter keys for
  a vault. `ragnarok backfill export` writes a JSON file with the title and the first lines of each public note
  that needs a description (credentials removed). Claude writes the descriptions into one or more JSON files,
  `{path: {"description": "...", "type": "..."}}`. `ragnarok backfill merge <vault> <file>...` puts them into the
  proposal, removes credentials, reports each description of more than 25 words and each row with no
  description, and replaces a default type with a valid type from Claude. You examine the proposal, then
  `ragnarok backfill apply` makes a backup archive of the vault and adds only the missing keys.
- **Evaluation.** `ragnarok eval` compares grep, keyword, dense and hybrid search on your own question sets.
  `ragnarok eval agent` compares a Claude Code agent with grep and an agent with the MCP tools.
- **One-command setup.** `./setup.sh` installs the tools, the model and the index, and registers the MCP server
  in Claude Code. `ragnarok doctor` checks the setup.

## Requirements

- macOS with [Homebrew](https://brew.sh). The setup uses `brew services` for Ollama.
- [Ollama](https://ollama.com): `brew install ollama`.
- [Claude Code](https://claude.com/claude-code).
- `setup.sh` installs [uv](https://docs.astral.sh/uv/) if it is missing. uv installs Python 3.12.

## Setup

```bash
git clone https://github.com/freyr/ragnarok.git
cd ragnarok
./setup.sh                      # installs uv if needed, then runs `ragnarok init`
```

`ragnarok init` does six steps. It skips each step that is already done:

1. It writes `config.toml` from `config.example.toml`. It asks for the vault root, the folder that holds your
   vault folders, and adds each subfolder as a vault.
2. It checks that Ollama is installed.
3. It runs Ollama as a login service with `brew services start ollama` (it asks first). If Ollama runs outside
   brew services, for example `ollama serve` in a terminal, init stops it and starts the service.
4. It pulls the embedding model.
5. It builds the index. A vault with 1,500 notes takes approximately 12 minutes on an Apple Silicon Mac.
6. It registers the MCP server `ragnarok` in Claude Code at the user scope (it asks before it replaces another
   registration).

Restart Claude Code after step 6. `uv run ragnarok doctor` checks the same six parts and prints the fix for
each failure.

Before step 5, examine `config.toml`: remove a folder that is not a vault, and add your private paths. A later
change of the configuration applies at the next `ragnarok index` or MCP tool call, with no restart.

## Use in Claude Code

The server gives five tools:

| Tool | What it does |
|---|---|
| `search` | Hybrid search over the notes. Filters: vault, project, area, type, archived. |
| `find_projects` | The projects that match a prompt, with the summary and the matching notes of each project. |
| `list_project` | All notes of one project. |
| `open_note` | The text of a note or of one section, with its links and backlinks. |
| `find_by_jira` | The notes that contain a ticket key, for example `ABC-123`. |

Claude Code loads MCP tools on demand. To make Claude use them before grep, add a short instruction to your
`~/.claude/CLAUDE.md` file, for example:

```markdown
## Finding notes: ragnarok first

The `ragnarok` MCP server indexes my Obsidian vaults. Use it before Grep or Glob whenever you look for
notes, earlier work or context:

- `find_projects`: the prompt continues or resembles earlier work and names no project. Then call
  `list_project` on the best match (look at the top 3).
- `search`: facts, decisions, how something was done. Filters: vault, project, area, type.
- `open_note`: read a note or one section, with its links and backlinks.
- `find_by_jira`: a ticket key such as `OPS-77`.

The tools are deferred. Load them in one call before the first use:
`ToolSearch` with `select:mcp__ragnarok__search,mcp__ragnarok__find_projects,mcp__ragnarok__list_project,mcp__ragnarok__open_note,mcp__ragnarok__find_by_jira`.

If the tools are absent (for example on a computer without ragnarok), fall back to Grep and say so.
Results can be unrelated: read the text before you rely on it, and cite note paths. Private notes are
refused by design; do not work around the refusal with Read or Grep.
```

## Adapting to your vaults

[docs/vault-requirements.md](docs/vault-requirements.md) tells how a vault must look to get the most from
ragnarok: headings, project folders, `SUMMARY.md`, frontmatter and private folders.

The default path rules expect `<vault>/<area>/<project>/note.md`. If your vaults look different, the repo
skill `.claude/skills/adapt-vault/` guides Claude Code through the changes. Open Claude Code in the checkout
and ask it to adapt ragnarok to your vaults. The skill:

1. runs `ragnarok survey` on each vault;
2. agrees the private folders with you, before any other change and before a new vault goes into `vaults`;
3. writes the path rules into `config.toml`, checked with `ragnarok inspect`;
4. runs the frontmatter backfill, if you agree, with your review before `apply`;
5. writes a small evaluation set from your notes and runs `ragnarok eval mechanical`;
6. adds the guidance above to your `~/.claude/CLAUDE.md`, after it asks you.

It changes only `config.toml` and adds frontmatter keys. It never moves, renames or deletes a note or a folder.

## Configuration

`config.toml` is in the checkout. A relative path in it is relative to the checkout. `config.example.toml`
explains each key. The most important keys:

| Key | What |
|---|---|
| `root`, `vaults` | The folder that holds the vaults, and the vault folders to index. |
| `private_patterns` | Glob patterns of private notes, for example `"Notes/Journal/**"`. |
| `model`, `dimension` | The Ollama embedding model and its vector size. A change needs `ragnarok index --rebuild`. |
| `min_words`, `max_words` | The chunk size limits of the splitter. A change needs `ragnarok index --rebuild`. |
| `project_markers` | A folder that holds one of these files is a project folder (default `DRIVERS.md`, `SUMMARY.md`). |
| `jira_prefixes` | The ticket prefixes that `find_by_jira` knows. |
| `[vault_rules.<vault>]` | The path rules of one vault: which folder gives the area and the project. |

### Note metadata

ragnarok reads these frontmatter keys. Each key is optional:

```yaml
---
project: heating-upgrade     # the project slug; default: from the path rules
area: home                   # a wide subject; default: from the path rules
type: plan                   # drivers, summary, research, spec, plan, brainstorm, review, decision,
                             # guide, reference, meeting, log, template or note
status: active               # active or archived; a note in an _archive folder, or in a project
                             # whose SUMMARY.md has stage: DONE, is archived
description: One sentence about the note, which improves find_projects.
jira: [ABC-123]              # ticket keys, in addition to the keys in the text
---
```

Without frontmatter, the path rules give the metadata. The default rules work for a vault with the layout
`<vault>/<area>/<project>/note.md`. For a different layout, add a `[vault_rules.<vault>]` table. Use
`ragnarok inspect <note path>` to see the metadata of a note.

## Commands

| Command | What it does |
|---|---|
| `ragnarok init` | Sets up this computer (see Setup). |
| `ragnarok doctor` | Checks the setup. |
| `ragnarok index [--rebuild]` | Adds, changes and removes notes in the index. After a change of the path rules, the private paths or the markers, it derives the metadata of all notes again, with no new embeddings. When a `.md` marker file is added or removed, it does the same for the notes under the folder of that file. `--rebuild` is needed only after a change of `model`, `dimension`, `min_words` or `max_words`. |
| `ragnarok status` | Shows the size and the state of the index. |
| `ragnarok search "<query>"` | Searches from the terminal. |
| `ragnarok inspect <path>...` | Shows the metadata of notes. |
| `ragnarok chunks <path>... \| --stats` | Shows the chunks of notes, or the totals. |
| `ragnarok serve` | Runs the MCP server on stdio. Claude Code starts it. |
| `ragnarok backfill propose\|export\|merge\|apply <vault> [file...]` | Proposes, describes and applies missing frontmatter keys. The order is `propose`, `export`, Claude writes descriptions, `merge <vault> <file>...`, you examine the proposal, `apply`. |
| `ragnarok survey <vault> [--json]` | Prints the folders, frontmatter keys, marker files, name patterns and ticket prefixes of a vault folder. Run it before you write `[vault_rules.<vault>]`. It prints no note text. |
| `ragnarok eval mechanical\|pool\|agent` | Runs an evaluation on the question sets in `data/eval/sets/`. |

Run each command with `uv run` in the checkout, for example `uv run ragnarok search "boiler"`.

## Where things are

Everything is in the checkout. Git ignores `config.toml`, `data/` and `logs/`.

| Path | What |
|---|---|
| `config.toml` | The configuration of this computer. |
| `data/index.db` | The SQLite index. Copy it to a second computer to skip the first index. |
| `data/backups/` | The vault archives that `backfill apply` makes before it writes. |
| `data/proposals/` | The output of `backfill propose` and `backfill export`. |
| `data/eval/` | Your question sets (`sets/`), the evaluation results and the mirror of the agent evaluation. |
| `logs/server.log` | The MCP server log, one line for each tool call with its time. |

Ollama keeps the model in `~/.ollama`. Set `RAGNAROK_HOME` to use a different home folder.

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## License

MIT. See [LICENSE](LICENSE).
