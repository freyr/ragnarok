# ragnarok

<!-- mcp-name: io.github.freyr/ragnarok -->

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
- **One-command setup.** `ragnarok init` installs the model and the index, and registers the MCP server in
  Claude Code when the Claude Code plugin does not give it. `ragnarok doctor` checks the setup.

## Requirements

- macOS with [Homebrew](https://brew.sh). The setup uses `brew services` for Ollama.
- [Ollama](https://ollama.com): `brew install ollama`.
- [Claude Code](https://claude.com/claude-code).
- [uv](https://docs.astral.sh/uv/): `brew install uv`. uv installs Python 3.12. `setup.sh` installs uv if it
  is missing.

## Setup

ragnarok is on [PyPI](https://pypi.org/p/ragnarok-mcp) as `ragnarok-mcp` and in the
[MCP Registry](https://registry.modelcontextprotocol.io) as `io.github.freyr/ragnarok`.

### With the Claude Code plugin (no clone)

In Claude Code:

```text
/plugin marketplace add freyr/ragnarok-claude-plugin
/plugin install ragnarok@ragnarok
```

Then in a terminal:

```bash
uvx ragnarok-mcp init
```

The plugin starts the server with `uvx ragnarok-mcp serve` and adds the search guidance to each session. The home
folder is `~/.ragnarok`. To use a different folder, set `RAGNAROK_HOME` in the `env` of
`~/.claude/settings.json` and in your shell.

### From a checkout

```bash
git clone https://github.com/freyr/ragnarok.git
cd ragnarok
./setup.sh                      # installs uv if needed, then runs `ragnarok init`
```

The checkout is the home folder.

### What init does

`ragnarok init` does six steps. It skips each step that is already done:

1. It writes `config.toml` from `config.example.toml`. It asks for the vault root, the folder that holds your
   vault folders, and adds each subfolder as a vault.
2. It checks that Ollama is installed.
3. It runs Ollama as a login service with `brew services start ollama` (it asks first). If Ollama runs outside
   brew services, for example `ollama serve` in a terminal, init stops it and starts the service.
4. It pulls the embedding model.
5. It builds the index. A vault with 1,500 notes takes approximately 12 minutes on an Apple Silicon Mac.
6. When the plugin `ragnarok` gives the MCP server, it removes a registration at the user scope (it asks
   first). Else it registers the MCP server `ragnarok` in Claude Code at the user scope (it asks before it
   replaces another registration).

Restart Claude Code after step 6. `ragnarok doctor` checks the same six parts and prints the fix for each
failure.

Before step 5, examine `config.toml`: remove a folder that is not a vault, and add your private paths. A later
change of the configuration applies at the next `ragnarok index` or MCP tool call, with no restart.

## Use in Claude Code

The server gives six tools:

| Tool | What it does |
|---|---|
| `search` | Hybrid search over the notes. Filters: vault, project, area, type, archived. |
| `find_projects` | The projects that match a prompt, with the summary and the matching notes of each project. |
| `list_project` | All notes of one project. |
| `open_projects` | The open projects (`state: in-progress` and `paused` by default), grouped by state, newest update first. |
| `open_note` | The text of a note or of one section, with its links and backlinks. A note of more than 4,000 words gives an outline: each section with its heading path, words and first sentence. |
| `find_by_jira` | The notes that contain a ticket key, for example `ABC-123`. |

The plugin adds the guidance below to each session itself, with the plugin tool names
`mcp__plugin_ragnarok_ragnarok__*`. Without the plugin, Claude Code loads MCP tools on demand. To make Claude
use them before grep, add a short instruction to your `~/.claude/CLAUDE.md` file, for example:

```markdown
## Finding notes: ragnarok first

The `ragnarok` MCP server indexes my Obsidian vaults. Use it before Grep or Glob whenever you look for
notes, earlier work or context:

- `find_projects`: the prompt continues or resembles earlier work and names no project. Then call
  `list_project` on the best match (look at the top 3).
- `search`: facts, decisions, how something was done. Filters: vault, project, area, type.
- `open_note`: read a note or one section, with its links and backlinks. A long note gives its outline first; open the section you need by heading.
- `open_projects`: which projects are in progress or paused.
- `find_by_jira`: a ticket key such as `OPS-77`.

The tools are deferred. Load them in one call before the first use:
`ToolSearch` with `select:mcp__ragnarok__search,mcp__ragnarok__find_projects,mcp__ragnarok__list_project,mcp__ragnarok__open_projects,mcp__ragnarok__open_note,mcp__ragnarok__find_by_jira`.

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

`config.toml` is in the home folder. A relative path in it is relative to the home folder.
`config.example.toml` explains each key. The most important keys:

| Key | What |
|---|---|
| `root`, `vaults` | The folder that holds the vaults, and the vault folders to index. |
| `[vault_paths]` | The folder of a vault that is not in `root`, for example `kb = "~/code/kb"`. A note key stays `<vault>/<path>`. |
| `private_patterns` | Glob patterns of private notes, for example `"Notes/Journal/**"`. |
| `model`, `dimension` | The Ollama embedding model and its vector size. A dimension below the native dimension of the model gives a shorter vector (Ollama `dimensions`). A change needs `ragnarok index --rebuild`. |
| `query_prefix`, `doc_template` | Override the query prefix and the document template of the model profile. `{title}` in the template is the note title and the heading path. The profiles cover `embeddinggemma-2:270m` (and `-bf16-text`), `qwen3-embedding:0.6b` and `:4b`, `snowflake-arctic-embed2` and `bge-m3`; another model gets the prefixes of `embeddinggemma-2:270m`. The old key `doc_prefix` still works as `doc_template`. A change needs `ragnarok index --rebuild`. |
| `keep_alive` | How long Ollama keeps the embedding model in memory after a call: an integer of seconds, `-1` (the default) keeps it loaded, or a duration such as `"30m"`. A change needs no rebuild. |
| `min_words`, `max_words` | The chunk size limits of the splitter. The splitter also cuts a chunk to the token limit of the model profile (4 characters for each token), and a text that Ollama refuses as too long is embedded in two halves; Ollama never truncates a text. A change needs `ragnarok index --rebuild`. |
| `chunk_prefix` | `none` (the default) or `description`: the description of a note goes before the text of each body chunk in the text to the model. The stored text and the keyword index do not change. A change needs `ragnarok index --rebuild`. |
| `note_rank` | How the chunks of a note give its rank: `maxp` (the default, the best chunk), `topk_sum` (the sum of the scores of the best 3 chunks of the note in the dense list and in the keyword list) or `note_vector` (also the mean chunk vector of each note, as a third list). A change needs a new index. |
| `[variants.<name>]` | The bench: a variant overrides `model`, `dimension`, `query_prefix`, `doc_template`, `min_words`, `max_words`, `chunk_prefix` and `note_rank`. |
| `[corpora.<name>]` | The bench: a `root` and a list of `vaults` that a variant indexes. |
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
                             # whose SUMMARY.md has state: superseded or abandoned, is archived; state: done is current
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
| `ragnarok index [--rebuild]` | Adds, changes and removes notes in the index. After a change of the path rules, the private paths or the markers, it derives the metadata of all notes again, with no new embeddings. When a `.md` marker file is added or removed, it does the same for the notes under the folder of that file. `--rebuild` is needed only after a change of `model`, `dimension`, `min_words` or `max_words`. It keeps the embedding cache, so it embeds only the chunk texts that are not in the cache. |
| `ragnarok status` | Shows the size and the state of the index. |
| `ragnarok search "<query>"` | Searches from the terminal. |
| `ragnarok inspect <path>...` | Shows the metadata of notes. |
| `ragnarok chunks <path>... \| --stats` | Shows the chunks of notes, or the totals. |
| `ragnarok serve` | Runs the MCP server on stdio. Claude Code starts it. |
| `ragnarok backfill propose\|export\|merge\|apply <vault> [file...]` | Proposes, describes and applies missing frontmatter keys. The order is `propose`, `export`, Claude writes descriptions, `merge <vault> <file>...`, you examine the proposal, `apply`. |
| `ragnarok survey <vault> [--json]` | Prints the folders, frontmatter keys, marker files, name patterns and ticket prefixes of a vault folder. Run it before you write `[vault_rules.<vault>]`. It prints no note text. |
| `ragnarok eval mechanical\|pool\|agent` | Runs an evaluation on the question sets in `data/eval/sets/`. |
| `ragnarok bench index --variant <name> --corpus <name>` | Builds or updates the index of a variant on a corpus in `data/variants/<variant>/<corpus>/index.db`. It shares the embedding cache, so a variant with the default embedding settings embeds nothing again. It refuses an index that was built with other settings of the variant. |

Run each command with `uvx ragnarok-mcp`, for example `uvx ragnarok-mcp search "boiler"`, or with `uv run` in the
checkout.

## Where things are

Everything is in the home folder: `RAGNAROK_HOME`, else the checkout, else `~/.ragnarok`. In a checkout, git
ignores `config.toml`, `data/` and `logs/`.

| Path | What |
|---|---|
| `config.toml` | The configuration of this computer. |
| `data/index.db` | The SQLite index. Copy it and `data/embed-cache.db` to a second computer to skip the first index. |
| `data/embed-cache.db` | The embedding cache that all index files share. Its key is the model, the dimension, the document template and the text hash. |
| `data/variants/<variant>/<corpus>/index.db` | The index of a bench variant on a corpus. The MCP server never uses it. |
| `data/backups/` | The vault archives that `backfill apply` makes before it writes. |
| `data/proposals/` | The output of `backfill propose` and `backfill export`. |
| `data/eval/` | Your question sets (`sets/`), the evaluation results and the mirror of the agent evaluation. |
| `logs/server.log` | The MCP server log, one line for each tool call with its time. |

Ollama keeps the model in `~/.ollama`.

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

### Release

1. Set the version in `pyproject.toml` and in both `version` fields of `server.json`, and add the section to
   `CHANGELOG.md`.
2. After the merge, push the tag: `git tag v1.2.3 && git push origin v1.2.3`.
3. The workflow `publish.yml` checks that the three versions match the tag, runs the tests, publishes the
   package to PyPI with trusted publishing, then publishes `server.json` to the MCP Registry.

## License

MIT. See [LICENSE](LICENSE).
