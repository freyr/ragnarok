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
| `ragnarok bench run --set <file> --variant <name>` | Runs each question of a set through the MCP search (`limit` 20, with the change scan, no filter) on the index of the variant on the corpus of the set. It writes the rank of each expected note, the scan time and the search time of each question, the keyword overlap of the question and the gold note, and the BM25 rank of the gold note to `data/bench/<set hash>-<variant hash>.json`. It first brings the index of the variant current with no limit. It refuses a set when an expected path is not in the index or is private, lists each such path with its reason and runs no question. |
| `ragnarok bench compare --set <file> <A> <B> [<B2>...] [--margin 0.05]` | Gives recall@5 and MRR of each variant, by stratum and by area, and the difference B − A with a 95% interval from a cluster bootstrap (10,000 samples, fixed seed). With more than one B, the interval level follows the Holm correction. It lists the share of discordant questions and each changed question with its diagnostics. It ends with one verdict line for each B (see the set format). |
| `ragnarok bench convert <old set>... --out <file> --name <name> --kind synthetic\|real --corpus <name> [--path-map <tsv>] [--area <area>]...` | Converts one or more old sets to the set format. It reads the draft format of `ragnarok eval` (`expected`, `type`) and the gold format of the health and KB sets (`gold` as a list of path and grade, `class`); each question shows its format. In the gold format, a gold path of grade 1 or more (no grade counts as 1) becomes an expected path, `class` becomes the stratum, the KB keys `gold_eligibility`, `gold_route`, `gold_claims` and `when` are dropped, and the other keys stay. In the draft format, the stratum is the old `type`. `--path-map` gives a TSV file with the columns `old` and `new`: each old expected path is replaced by the new one. The cluster is the first expected path that is a note, else the first expected path, and the area comes from the metadata rules of the cluster note (`unknown` when the note is absent). It does not convert a question with no expected path, with a folder as an expected path, of an area that `--area` does not select, or with the text of an earlier question, and it lists each one. An id that an earlier question has gets the prefix `<stratum>-`. |

### The set format

A set is one JSON file:

```json
{
  "name": "pilot", "kind": "real", "version": 2, "corpus": "real", "hash": "<SHA-256 of the questions>",
  "questions": [
    {"id": "q01", "question": "What is a field stop?", "expected": ["Private/Astro/Field Stop.md"],
     "stratum": "lookup", "area": "astro", "cluster": "Private/Astro/Field Stop.md", "notes": "optional"}
  ]
}
```

- `kind` is `synthetic` or `real`. `corpus` is a table `[corpora.<name>]` of the config.
- `hash` is the SHA-256 of the `questions` list as JSON with sorted keys and no spaces. The bench refuses a set whose hash does not agree.
- `expected` holds one or more note paths. A continuation question lists the `SUMMARY.md` of its project.
- `cluster` is the path of the main gold note. The bootstrap draws clusters, not questions.
- recall@5 is 1 when all expected notes are in the first 5 notes, else 0. MRR uses the rank of the first expected note; no expected note in the first 20 gives 0.

The verdict of `bench compare`:

- A synthetic set: B wins when the interval of B − A at its Holm level is above 0 for recall@5 or for MRR, the point difference of the other measure is not below 0, and the p95 search time (scan included) is below 500 ms. The report also gives the sign of the difference in each vault.
- A real set does not decide a win. It checks that B does not get worse: the lower end of the interval of B − A must be above −margin for recall@5 and for MRR. It also checks the p95 time.

### The synthetic set

`ragnarok synth` makes a synthetic vault with known answers in a work folder, one folder for each seed. Run the
commands in a checkout: they read the prompts in `bench/synthetic/prompts/` and call `claude -p` (no tools, no
settings, no `CLAUDE.md`).

| Command | What it does |
|---|---|
| `ragnarok synth graph --seed <n> --questions <n> --out <folder>` | Builds the seeded fact graph of the vault `seed<n>`: areas, projects with `DRIVERS.md` and `SUMMARY.md`, devices, places, people and their facts, each with a unique value in exactly one current note, plus distractor notes, archived notes with an older value and long notes (3,000-10,000 words) with the fact in the last section. It plans one question for each gold note, in the strata `direct` 20%, `lexical-gap` 20%, `deep-section` 20%, `distractor` 15%, `outdated` 10%, `continuation` 10% and `multi-hop` 5%, each with a `dev` or `test` half and a property set that matches exactly one subject. It writes `world.json` and `graph.json` and prints the graph hash; the same seed gives the same hash. |
| `ragnarok synth render <folder> [--model sonnet] [--jobs 4]` | Writes each note with `claude -p` from its facts, the properties of its subject and the style targets (aggregate numbers only) into `<folder>/vault/seed<n>/`. A note that fails its string checks is written again at most twice and then removed with its question. A second run continues. |
| `ragnarok synth questions <folder> [--model sonnet] [--jobs 4]` | Writes each question from its fact tuple only, with the forbidden words and the instruction of its stratum, and gives it to a blind `claude -p` call with the subjects of the same class and area and the subjects of the same class in other areas that differ from its property set in at most one property. It refuses to run until each note is rendered. A question with a forbidden word or that the blind call does not resolve to its subject is written again at most twice and then removed. |
| `ragnarok synth verify <folder>` | Checks the graph hash, the property sets, the strata and the halves, that each value is in its note and in no other note (a distractor holds no gold value), that each note states the property values that its questions use (a sibling note its own values), the length of each long note and the position of its fact, the forbidden words and the answer of the blind check. It lists each failed check, writes `verify.json` and fails when a kept note or question fails. |
| `ragnarok synth freeze <folder>... --name <set> [--out bench/synthetic/<set>]` | Freezes verified work folders into one set folder: `vault.tar.gz`, `questions.json` (a set of kind `synthetic` whose corpus is the set name, with the `half` of each question), the `world-` and `graph-` file of each seed, the prompts, `verify.json` and `manifest.json` with the SHA-256 of each file, the model IDs and the SHA-256 of each prompt. Each note and question records the SHA-256 of the prompts that wrote it, and freeze refuses when they differ from the prompts that it copies. It refuses when a check fails or when the notes, the questions or the graphs hold a term of the deny list. |

The deny list holds the terms that must never be in a synthetic set: vault names, real names, real paths. It stays
outside the repo, in `RAGNAROK_SYNTH_DENY`, else `data/synth-deny.txt` of the home folder: one term on each line, `#`
starts a comment, no case, whole words for a term of letters. The tests marked `local` call `claude -p` and read the
deny list; CI skips them, `uv run pytest -m local` runs them.

To measure a frozen set, unpack `vault.tar.gz` into a folder and add `[corpora.<set>]` with that folder as `root` and
the vaults `seed<n>` of the manifest.

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
| `data/bench/` | The results of `bench run`, one file for each set and variant settings. |
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
